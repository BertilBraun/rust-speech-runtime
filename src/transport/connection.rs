use std::{ops::ControlFlow, time::Duration};

use tokio::{net::TcpStream, time::Instant};
use tokio_util::sync::CancellationToken;

use super::{
    gateway::{GatewayConfig, GatewayError},
    wire::{ClientRequest, ServerPeer, ServerReply, WireError},
};
use crate::{
    Ingress,
    config::RuntimeConfig,
    protocol::{
        AudioPacket, CreateOutcome, FrameRejection, InputFrame, InputOutcome, SessionId,
        SessionLease,
    },
};

enum AudioExchange {
    Completed(InputOutcome),
    Interrupted(Option<ClientRequest>),
    Cancelled,
}

pub(super) struct Connection {
    peer: ServerPeer,
    ingress: Ingress,
    runtime: RuntimeConfig,
    configuration: GatewayConfig,
    cancellation: CancellationToken,
    lease: Option<SessionLease>,
}

impl Connection {
    pub(super) fn new(
        stream: TcpStream,
        ingress: Ingress,
        runtime: RuntimeConfig,
        configuration: GatewayConfig,
        cancellation: CancellationToken,
    ) -> Result<Self, GatewayError> {
        Ok(Self {
            peer: ServerPeer::new(stream, configuration.message_limit)?,
            ingress,
            runtime,
            configuration,
            cancellation,
            lease: None,
        })
    }

    pub(super) async fn run(mut self) -> Result<(), GatewayError> {
        let outcome = self.serve_requests().await;
        self.close_session().await?;
        outcome
    }

    async fn serve_requests(&mut self) -> Result<(), GatewayError> {
        while let Some(request) = self.receive().await? {
            if self.handle_request(request).await?.is_break() {
                break;
            }
        }
        Ok(())
    }

    async fn receive(&mut self) -> Result<Option<ClientRequest>, GatewayError> {
        tokio::select! {
            _ = self.cancellation.cancelled() => Ok(None),
            outcome = tokio::time::timeout(self.configuration.io_timeout, self.peer.receive()) => {
                Ok(outcome.map_err(|_| WireError::Timeout)??)
            }
        }
    }

    async fn send_reply(&mut self, reply: ServerReply) -> Result<(), GatewayError> {
        tokio::select! {
            _ = self.cancellation.cancelled() => Ok(()),
            outcome = tokio::time::timeout(self.configuration.io_timeout, self.peer.send(reply)) => {
                outcome.map_err(|_| WireError::Timeout)??;
                Ok(())
            }
        }
    }

    async fn handle_request(
        &mut self,
        request: ClientRequest,
    ) -> Result<ControlFlow<()>, GatewayError> {
        match request {
            ClientRequest::Open(session_id) => self.open_session(session_id).await,
            ClientRequest::Audio {
                packet,
                remaining_budget,
            } => self.audio(packet, remaining_budget).await,
            ClientRequest::EvictCache => {
                let lease = self.require_session("open a session before eviction")?;
                let removed = self.ingress.evict_cache(lease).await?;
                self.send_reply(ServerReply::CacheEvicted(removed)).await?;
                Ok(ControlFlow::Continue(()))
            }
            ClientRequest::Close => {
                let closed = self.close_session().await?;
                self.send_reply(ServerReply::Closed(closed)).await?;
                Ok(ControlFlow::Break(()))
            }
        }
    }

    async fn open_session(
        &mut self,
        session_id: SessionId,
    ) -> Result<ControlFlow<()>, GatewayError> {
        if self.lease.is_some() {
            return Err(WireError::InvalidMessage("connection already owns a session").into());
        }
        let outcome = self.ingress.create_session(session_id).await?;
        if let CreateOutcome::Admitted(admission) = outcome {
            self.lease = Some(SessionLease {
                session_id,
                generation: admission.assignment.generation,
            });
        }
        self.send_reply(ServerReply::Opened(outcome)).await?;
        Ok(if self.lease.is_some() {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        })
    }

    fn require_session(&self, message: &'static str) -> Result<SessionLease, WireError> {
        self.lease.ok_or(WireError::InvalidMessage(message))
    }

    async fn close_session(&mut self) -> Result<bool, GatewayError> {
        match self.lease.take() {
            Some(lease) => Ok(self.ingress.close_session(lease).await?),
            None => Ok(false),
        }
    }

    async fn audio(
        &mut self,
        packet: AudioPacket,
        remaining_budget: Duration,
    ) -> Result<ControlFlow<()>, GatewayError> {
        let lease = self.require_session("open a session before audio")?;
        if remaining_budget.is_zero() || remaining_budget > self.runtime.packet_recovery_budget() {
            self.send_reply(ServerReply::Rejected(FrameRejection::DeadlineExceeded))
                .await?;
            return Ok(ControlFlow::Break(()));
        }
        let timestamp = Instant::now();
        // Recovery preserves input state; EDF retains the original playback target.
        let input = InputFrame {
            timestamp,
            deadline: timestamp + remaining_budget
                - (self.runtime.packet_recovery_budget() - self.runtime.packet_deadline),
            packet,
        };
        let outcome = match self.wait_for_audio(lease, input).await? {
            AudioExchange::Completed(outcome) => outcome,
            AudioExchange::Interrupted(request) => {
                self.interrupt_audio(request).await?;
                return Ok(ControlFlow::Break(()));
            }
            AudioExchange::Cancelled => return Ok(ControlFlow::Break(())),
        };
        let terminal = matches!(outcome, InputOutcome::Rejected(_));
        let reply = audio_reply(outcome);
        self.send_reply(reply).await?;
        Ok(if terminal {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        })
    }

    async fn wait_for_audio(
        &mut self,
        lease: SessionLease,
        input: InputFrame,
    ) -> Result<AudioExchange, GatewayError> {
        tokio::select! {
            _ = self.cancellation.cancelled() => Ok(AudioExchange::Cancelled),
            outcome = self.ingress.input_frame(lease, input) => {
                Ok(AudioExchange::Completed(outcome?))
            }
            incoming = self.peer.receive() => Ok(AudioExchange::Interrupted(incoming?)),
        }
    }

    async fn interrupt_audio(
        &mut self,
        request: Option<ClientRequest>,
    ) -> Result<(), GatewayError> {
        let reply = match request {
            None => return Ok(()),
            Some(ClientRequest::Close) => ServerReply::Closed(self.close_session().await?),
            Some(_) => {
                self.close_session().await?;
                ServerReply::Rejected(FrameRejection::Overloaded)
            }
        };
        self.send_reply(reply).await
    }
}

fn audio_reply(outcome: InputOutcome) -> ServerReply {
    match outcome {
        InputOutcome::Processed(mut output) => {
            output.audio.timings.gateway_return = Instant::now()
                .duration_since(output.completed_at)
                .saturating_sub(output.audio.timings.result_delivery);
            ServerReply::Audio(output.audio)
        }
        InputOutcome::CacheMiss => ServerReply::CacheMiss,
        InputOutcome::Rejected(reason) => ServerReply::Rejected(reason),
    }
}
