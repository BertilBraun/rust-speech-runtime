use super::FailureReason;
use crate::{
    metrics::{
        LatencyHistogram,
        profile::{PacketTimings, RETAINED_TRACES},
    },
    protocol::{AudioResult, PacketSequence, SessionId},
};
use serde::{Deserialize, Serialize};
use std::{cmp::Reverse, time::Duration};
use tokio::sync::mpsc;

#[derive(Debug, Serialize, Deserialize)]
pub struct ClientEchoTrace {
    pub session_id: SessionId,
    pub sequence: PacketSequence,
    pub round_trip: Duration,
    pub lateness: Duration,
    pub client_start_delay: Duration,
    pub outside_server: Duration,
    pub server: PacketTimings,
}
impl ClientEchoTrace {
    pub(super) fn new(
        session_id: SessionId,
        audio: &AudioResult,
        round_trip: Duration,
        client_start_delay: Duration,
        packet_deadline: Duration,
    ) -> Self {
        Self {
            session_id,
            sequence: audio.sequence,
            round_trip,
            lateness: round_trip.saturating_sub(packet_deadline),
            client_start_delay,
            outside_server: round_trip.saturating_sub(client_start_delay + audio.timings.total()),
            server: *audio.timings,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ClientPacketTrace {
    Echo(ClientEchoTrace),
    LateEcho(ClientEchoTrace),
    Failure {
        session_id: SessionId,
        sequence: PacketSequence,
        round_trip: Duration,
        client_start_delay: Duration,
        reason: FailureReason,
    },
}
impl ClientPacketTrace {
    fn round_trip(&self) -> Duration {
        match self {
            Self::Echo(echo) | Self::LateEcho(echo) => echo.round_trip,
            Self::Failure { round_trip, .. } => *round_trip,
        }
    }
}

#[derive(Default)]
pub(super) struct PacketMeasurements {
    pub round_trip: LatencyHistogram,
    pub failed_round_trip: LatencyHistogram,
    pub deadline_lateness: LatencyHistogram,
    pub client_start_delay: LatencyHistogram,
    pub outside_server: LatencyHistogram,
    pub slowest_packets: Vec<ClientPacketTrace>,
}
impl PacketMeasurements {
    pub(super) async fn collect(mut self, mut events: mpsc::Receiver<ClientPacketTrace>) -> Self {
        while let Some(packet) = events.recv().await {
            match &packet {
                ClientPacketTrace::Echo(echo) => {
                    self.round_trip.record(echo.round_trip);
                    self.client_start_delay.record(echo.client_start_delay);
                    self.outside_server.record(echo.outside_server);
                }
                ClientPacketTrace::LateEcho(echo) => {
                    self.round_trip.record(echo.round_trip);
                    self.failed_round_trip.record(echo.round_trip);
                    self.deadline_lateness.record(echo.lateness);
                    self.client_start_delay.record(echo.client_start_delay);
                    self.outside_server.record(echo.outside_server);
                }
                ClientPacketTrace::Failure {
                    round_trip,
                    client_start_delay,
                    ..
                } => {
                    self.failed_round_trip.record(*round_trip);
                    self.client_start_delay.record(*client_start_delay);
                }
            }
            self.slowest_packets.push(packet);
            self.slowest_packets
                .sort_unstable_by_key(|packet| Reverse(packet.round_trip()));
            self.slowest_packets.truncate(RETAINED_TRACES);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{ClientEchoTrace, ClientPacketTrace, PacketMeasurements};
    use crate::{
        metrics::profile::PacketTimings,
        protocol::{PacketSequence, SessionId},
    };
    use std::time::Duration;
    use tokio::sync::mpsc;

    fn echo(round_trip_ms: u64) -> ClientEchoTrace {
        ClientEchoTrace {
            session_id: SessionId(1),
            sequence: PacketSequence(0),
            round_trip: Duration::from_millis(round_trip_ms),
            lateness: Duration::from_millis(round_trip_ms.saturating_sub(50)),
            client_start_delay: Duration::from_millis(1),
            outside_server: Duration::from_millis(round_trip_ms - 13),
            server: PacketTimings {
                device_execution: Duration::from_millis(12),
                ..PacketTimings::default()
            },
        }
    }

    #[tokio::test]
    async fn late_echo_is_counted_in_latency_and_lateness_statistics() {
        let (events, mailbox) = mpsc::channel(2);
        events
            .send(ClientPacketTrace::Echo(echo(20)))
            .await
            .unwrap();
        events
            .send(ClientPacketTrace::LateEcho(echo(51)))
            .await
            .unwrap();
        drop(events);
        let measurements = PacketMeasurements::default().collect(mailbox).await;
        assert_eq!(measurements.round_trip.summary().samples, 2);
        assert_eq!(measurements.deadline_lateness.summary().samples, 1);
        assert_eq!(measurements.failed_round_trip.summary().samples, 1);
        assert_eq!(measurements.client_start_delay.summary().samples, 2);
        assert_eq!(measurements.outside_server.summary().samples, 2);
        let ClientPacketTrace::LateEcho(trace) = &measurements.slowest_packets[0] else {
            panic!("late echo must remain the slowest trace");
        };
        assert_eq!(trace.server.device_execution, Duration::from_millis(12));
        assert_eq!(trace.outside_server, Duration::from_millis(38));
    }
}
