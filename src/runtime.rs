use crate::{
    config::{ConfigError, RuntimeConfig},
    metrics::Report,
    protocol::{CreateOutcome, FrameRejection, InputFrame, InputOutcome, SessionId, SessionTarget},
    session::manager::SessionManager,
};
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
        target: SessionTarget,
        input: InputFrame,
        reply: oneshot::Sender<InputOutcome>,
    },
    CloseSession {
        target: SessionTarget,
        reply: oneshot::Sender<bool>,
    },
    EvictCache {
        target: SessionTarget,
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
        target: impl Into<SessionTarget>,
        input: InputFrame,
    ) -> Result<InputOutcome, RuntimeError> {
        if self.cancellation.is_cancelled() {
            return Err(RuntimeError::Stopped);
        }
        let target = target.into();
        if input.packet.payload.len() > self.configuration.audio_limits.max_frame_bytes {
            return Err(RuntimeError::InvalidFrame(
                "payload exceeds max_frame_bytes",
            ));
        }
        if input.timestamp > Instant::now()
            || input.deadline > input.timestamp + self.configuration.packet_deadline
        {
            return Err(RuntimeError::InvalidFrame(
                "timestamp or deadline is outside the allowed budget",
            ));
        }
        let (reply, response) = oneshot::channel();
        match self.commands.try_send(Command::InputFrame {
            target,
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
                self.close_session(target).await?;
                Ok(InputOutcome::Rejected(FrameRejection::Overloaded))
            }
            Err(mpsc::error::TrySendError::Closed(_)) => Err(RuntimeError::Stopped),
        }
    }
    pub async fn close_session(
        &self,
        target: impl Into<SessionTarget>,
    ) -> Result<bool, RuntimeError> {
        let (reply, response) = oneshot::channel();
        self.send_control(Command::CloseSession {
            target: target.into(),
            reply,
        })
        .await?;
        response.await.map_err(|_| RuntimeError::Stopped)
    }
    pub async fn evict_cache(
        &self,
        target: impl Into<SessionTarget>,
    ) -> Result<bool, RuntimeError> {
        let (reply, response) = oneshot::channel();
        self.send_control(Command::EvictCache {
            target: target.into(),
            reply,
        })
        .await?;
        response.await.map_err(|_| RuntimeError::Stopped)
    }
    async fn send_control(&self, command: Command) -> Result<(), RuntimeError> {
        if self.cancellation.is_cancelled() {
            return Err(RuntimeError::Stopped);
        }
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
    cancellation: CancellationToken,
    manager: Option<JoinHandle<Result<Report, RuntimeError>>>,
}
impl Node {
    pub async fn start(configuration: RuntimeConfig) -> Result<Self, RuntimeError> {
        configuration.validate()?;
        let configuration = Arc::new(configuration);
        let (commands, mailbox) = mpsc::channel(configuration.ingress_capacity);
        let cancellation = CancellationToken::new();
        let measurements = Arc::new(IngressMeasurements::default());
        let ingress = Ingress {
            commands,
            configuration: configuration.clone(),
            measurements: measurements.clone(),
            cancellation: cancellation.clone(),
        };
        let manager = SessionManager::new(configuration, measurements).await?;
        let task = tokio::spawn(manager.run(mailbox, cancellation.clone()));
        Ok(Self {
            ingress,
            cancellation,
            manager: Some(task),
        })
    }
    pub async fn shutdown(mut self) -> Result<Report, RuntimeError> {
        self.cancellation.cancel();
        self.manager
            .take()
            .expect("manager remains owned until shutdown")
            .await?
    }
}
impl Drop for Node {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}
