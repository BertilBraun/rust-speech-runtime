use super::{
    ClientCounters, ClientEchoTrace, ClientPacketTrace, FailureReason, SimulationConfig,
    quality::PacketQuality,
};
use crate::{
    protocol::{CacheOutcome, PacketSequence, SessionId},
    transport::{AudioDelivery, AudioSession, ClientError, ConnectOutcome, connect_session},
};
use bytes::Bytes;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::{net::SocketAddr, time::Duration};
use tokio::{
    sync::{mpsc, oneshot},
    time::Instant,
};
use tokio_util::sync::CancellationToken;

pub(super) struct SessionInput {
    pub(super) ticks: mpsc::Receiver<Instant>,
    pub(super) cancellation: CancellationToken,
    pub(super) ready: oneshot::Sender<()>,
}

pub(super) fn classify(error: ClientError) -> FailureReason {
    match error {
        ClientError::DeadlineExceeded | ClientError::RecoveryExceeded(_) => {
            FailureReason::DeadlineExceeded
        }
        ClientError::Rejected(reason) => FailureReason::ServerRejected(reason),
        ClientError::Wire(_) => FailureReason::Connection,
        ClientError::Protocol(_) => FailureReason::Protocol,
        ClientError::PrefixCapacity => FailureReason::PrefixCapacity,
        ClientError::EchoMismatch => FailureReason::InvalidEcho,
    }
}

pub(super) struct SimulatedSession {
    session: Box<AudioSession>,
    session_id: SessionId,
    address: SocketAddr,
    configuration: SimulationConfig,
    metrics: mpsc::Sender<ClientPacketTrace>,
    payload: Bytes,
    counters: ClientCounters,
    quality: PacketQuality,
    random: SmallRng,
    created_at: Instant,
    lifetime: Option<Duration>,
}

impl SimulatedSession {
    pub(super) fn new(
        session: Box<AudioSession>,
        session_id: SessionId,
        address: SocketAddr,
        configuration: SimulationConfig,
        metrics: mpsc::Sender<ClientPacketTrace>,
    ) -> Self {
        let mut random = SmallRng::seed_from_u64(configuration.seed.wrapping_add(session_id.0));
        let lifetime = configuration
            .churn_after
            .map(|duration| duration.mul_f64(random.random_range(0.5..=1.0)));
        Self {
            session,
            session_id,
            address,
            metrics,
            payload: Bytes::from(vec![42; configuration.payload_bytes]),
            counters: ClientCounters {
                admitted_sessions: 1,
                ..ClientCounters::default()
            },
            quality: PacketQuality::new(configuration.quality),
            configuration,
            random,
            created_at: Instant::now(),
            lifetime,
        }
    }

    pub(super) async fn run(mut self, input: SessionInput) -> ClientCounters {
        let SessionInput {
            mut ticks,
            cancellation,
            ready,
        } = input;
        self.created_at = Instant::now();
        let _ = ready.send(());
        loop {
            let captured_at = tokio::select! {
                _ = cancellation.cancelled() => {
                    self.counters.failure(FailureReason::GeneratorOverrun);
                    return self.counters;
                }
                tick = ticks.recv() => match tick {
                    Some(tick) => tick,
                    None => break,
                },
            };
            let sequence = match self.infer_packet(captured_at).await {
                Ok(sequence) => sequence,
                Err(reason) => {
                    self.counters.failure(reason);
                    if reason == FailureReason::DeadlineMissBurst
                        && matches!(
                            tokio::time::timeout(
                                self.configuration.io_timeout,
                                self.session.close()
                            )
                            .await,
                            Ok(Ok(_))
                        )
                    {
                        self.counters.closed_sessions += 1;
                    }
                    return self.counters;
                }
            };
            if let Err(reason) = self.evict_if_due(sequence).await {
                self.counters.failure(reason);
                return self.counters;
            }
            if self
                .lifetime
                .is_some_and(|duration| self.created_at.elapsed() >= duration)
            {
                match tokio::time::timeout(self.configuration.io_timeout, self.session.close())
                    .await
                {
                    Ok(Ok(_)) => self.counters.closed_sessions += 1,
                    _ => {
                        self.counters.failure(FailureReason::Connection);
                        return self.counters;
                    }
                }
                match connect_session(self.address, self.session_id, self.configuration.io_timeout)
                    .await
                {
                    Ok(ConnectOutcome::Admitted(session)) => {
                        self.session = session;
                        self.counters.admitted_sessions += 1;
                        self.quality = PacketQuality::new(self.configuration.quality);
                    }
                    Ok(ConnectOutcome::Rejected(reason)) => {
                        self.counters.admission(reason);
                        return self.counters;
                    }
                    Err(error) => {
                        self.counters.failure(classify(error));
                        return self.counters;
                    }
                }
                self.created_at = Instant::now();
                self.lifetime = self
                    .configuration
                    .churn_after
                    .map(|duration| duration.mul_f64(self.random.random_range(0.5..=1.0)));
            }
        }
        match tokio::time::timeout(self.configuration.io_timeout, self.session.close()).await {
            Ok(Ok(_)) => self.counters.closed_sessions += 1,
            _ => self.counters.failure(FailureReason::Connection),
        }
        self.counters
    }

    async fn infer_packet(
        &mut self,
        captured_at: Instant,
    ) -> Result<PacketSequence, FailureReason> {
        self.counters.attempted_frames += 1;
        let sequence = self.session.next_sequence();
        let client_start_delay = captured_at.elapsed();
        let outcome = self
            .session
            .infer_audio(self.payload.clone(), captured_at)
            .await;
        let round_trip = captured_at.elapsed();
        let outcome = outcome.and_then(|delivery| {
            self.session
                .classify_echo(delivery.into_audio(), round_trip)
        });
        let delivery = match outcome {
            Ok(delivery) => delivery,
            Err(error) => {
                let (reason, trace) = match error {
                    ClientError::RecoveryExceeded(audio) => (
                        FailureReason::DeadlineExceeded,
                        ClientPacketTrace::UnrecoveredEcho(ClientEchoTrace::new(
                            self.session_id,
                            &audio,
                            round_trip,
                            client_start_delay,
                            self.session.packet_deadline(),
                        )),
                    ),
                    error => {
                        let reason = classify(error);
                        (
                            reason,
                            ClientPacketTrace::Failure {
                                session_id: self.session_id,
                                sequence,
                                round_trip,
                                client_start_delay,
                                reason,
                            },
                        )
                    }
                };
                self.emit(trace);
                return Err(reason);
            }
        };
        self.record_delivery(delivery, round_trip, client_start_delay)
    }

    fn record_delivery(
        &mut self,
        delivery: AudioDelivery,
        round_trip: Duration,
        client_start_delay: Duration,
    ) -> Result<PacketSequence, FailureReason> {
        let late = !matches!(delivery, AudioDelivery::OnTime(_));
        let discarded = matches!(delivery, AudioDelivery::Discarded(_));
        let audio = delivery.into_audio();
        let trace = ClientEchoTrace::new(
            self.session_id,
            &audio,
            round_trip,
            client_start_delay,
            self.session.packet_deadline(),
        );
        let trace = if discarded {
            self.counters.discarded_output_frames += 1;
            ClientPacketTrace::ExpiredEcho(trace)
        } else if late {
            ClientPacketTrace::LateEcho(trace)
        } else {
            ClientPacketTrace::Echo(trace)
        };
        self.emit(trace);
        self.counters.echoed_frames += 1;
        self.counters.late_frames += u64::from(late);
        let burst = self.quality.observe(late);
        self.counters.maximum_consecutive_misses = self
            .counters
            .maximum_consecutive_misses
            .max(self.quality.consecutive_misses);
        self.counters.maximum_window_misses = self
            .counters
            .maximum_window_misses
            .max(self.quality.window_misses);
        match audio.cache {
            CacheOutcome::Hit => self.counters.cache_hits += 1,
            CacheOutcome::Replayed { packets, bytes } => {
                self.counters.replayed_frames += 1;
                self.counters.replayed_packets += packets;
                self.counters.replayed_bytes += bytes;
            }
        }
        if burst {
            self.counters.quality_failed_sessions += 1;
            Err(FailureReason::DeadlineMissBurst)
        } else {
            Ok(audio.sequence)
        }
    }

    async fn evict_if_due(&mut self, sequence: PacketSequence) -> Result<(), FailureReason> {
        if !self
            .configuration
            .evict_every
            .is_some_and(|period| (sequence.0 + 1).is_multiple_of(period))
        {
            return Ok(());
        }
        match tokio::time::timeout(self.configuration.io_timeout, self.session.evict_cache()).await
        {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(error)) => Err(classify(error)),
            Err(_) => Err(FailureReason::Connection),
        }
    }

    fn emit(&mut self, trace: ClientPacketTrace) {
        if self.metrics.try_send(trace).is_err() {
            self.counters.metric_samples_dropped += 1;
        }
    }
}
