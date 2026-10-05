use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
    time::Instant,
};
use tokio_util::sync::CancellationToken;

use crate::config::{ConfigError, RuntimeConfig};
use crate::metrics::Report;
use crate::protocol::{CreateOutcome, InferenceOutput, InputFrame, InputOutcome, SessionId};
use crate::scheduler::placement::{LeastLoaded, PlacementPolicy};
use crate::session::manager::SessionManager;

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Configuration(#[from] ConfigError),
    #[error("runtime has stopped")]
    Stopped,
    #[error("invalid input frame: {0}")]
    InvalidFrame(&'static str),
    #[error(transparent)]
    Task(#[from] tokio::task::JoinError),
}

pub(crate) enum Command {
    CreateSession {
        session_id: SessionId,
        reply: oneshot::Sender<CreateOutcome>,
    },
    InputFrame {
        session_id: SessionId,
        input: InputFrame,
        reply: oneshot::Sender<InputOutcome>,
    },
    CloseSession {
        session_id: SessionId,
        reply: oneshot::Sender<bool>,
    },
}

#[derive(Default)]
pub(crate) struct IngressMeasurements {
    pub channel_saturation: AtomicU64,
    pub inputs_overloaded: AtomicU64,
}

#[derive(Clone)]
pub struct Ingress {
    commands: mpsc::Sender<Command>,
    configuration: Arc<RuntimeConfig>,
    measurements: Arc<IngressMeasurements>,
    cancellation: CancellationToken,
}

impl Ingress {
    pub async fn create_session(
        &self,
        session_id: SessionId,
    ) -> Result<CreateOutcome, RuntimeError> {
        let (reply, response) = oneshot::channel();
        self.send_control(Command::CreateSession { session_id, reply })
            .await?;
        response.await.map_err(|_| RuntimeError::Stopped)
    }

    pub async fn input_frame(
        &self,
        session_id: SessionId,
        input: InputFrame,
    ) -> Result<InputOutcome, RuntimeError> {
        if input.payload.len() > self.configuration.max_frame_bytes {
            return Err(RuntimeError::InvalidFrame(
                "payload exceeds max_frame_bytes",
            ));
        }
        if input.timestamp > Instant::now() {
            return Err(RuntimeError::InvalidFrame("timestamp is in the future"));
        }
        let (reply, response) = oneshot::channel();
        match self.commands.try_send(Command::InputFrame {
            session_id,
            input,
            reply,
        }) {
            Ok(()) => response.await.map_err(|_| RuntimeError::Stopped),
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.measurements
                    .channel_saturation
                    .fetch_add(1, Ordering::Relaxed);
                self.measurements
                    .inputs_overloaded
                    .fetch_add(1, Ordering::Relaxed);
                Ok(InputOutcome::Overloaded)
            }
            Err(mpsc::error::TrySendError::Closed(_)) => Err(RuntimeError::Stopped),
        }
    }

    pub async fn close_session(&self, session_id: SessionId) -> Result<bool, RuntimeError> {
        let (reply, response) = oneshot::channel();
        self.send_control(Command::CloseSession { session_id, reply })
            .await?;
        response.await.map_err(|_| RuntimeError::Stopped)
    }

    async fn send_control(&self, command: Command) -> Result<(), RuntimeError> {
        match self.commands.try_send(command) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(command)) => {
                self.measurements
                    .channel_saturation
                    .fetch_add(1, Ordering::Relaxed);
                tokio::select! {
                    _ = self.cancellation.cancelled() => Err(RuntimeError::Stopped),
                    result = self.commands.send(command) => result.map_err(|_| RuntimeError::Stopped),
                }
            }
            Err(mpsc::error::TrySendError::Closed(_)) => Err(RuntimeError::Stopped),
        }
    }
}

pub struct Node {
    pub ingress: Ingress,
    outputs: Option<mpsc::Receiver<InferenceOutput>>,
    cancellation: CancellationToken,
    manager: Option<JoinHandle<Result<Report, RuntimeError>>>,
}

impl Node {
    pub fn start(configuration: RuntimeConfig) -> Result<Self, RuntimeError> {
        Self::start_with_policy(configuration, Box::new(LeastLoaded))
    }

    pub fn start_with_policy(
        configuration: RuntimeConfig,
        placement: Box<dyn PlacementPolicy>,
    ) -> Result<Self, RuntimeError> {
        configuration.validate()?;
        let configuration = Arc::new(configuration);
        let (commands, mailbox) = mpsc::channel(configuration.ingress_capacity);
        let (outputs, output_mailbox) = mpsc::channel(configuration.output_channel_capacity);
        let cancellation = CancellationToken::new();
        let measurements = Arc::new(IngressMeasurements::default());
        let ingress = Ingress {
            commands,
            configuration: configuration.clone(),
            measurements: measurements.clone(),
            cancellation: cancellation.clone(),
        };
        let manager = SessionManager::new(configuration, placement, outputs, measurements);
        let task = tokio::spawn(manager.run(mailbox, cancellation.clone()));
        Ok(Self {
            ingress,
            outputs: Some(output_mailbox),
            cancellation,
            manager: Some(task),
        })
    }

    pub fn take_outputs(&mut self) -> Option<mpsc::Receiver<InferenceOutput>> {
        self.outputs.take()
    }

    pub async fn shutdown(mut self) -> Result<Report, RuntimeError> {
        self.cancellation.cancel();
        self.manager
            .take()
            .expect("manager is owned until shutdown")
            .await?
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}
