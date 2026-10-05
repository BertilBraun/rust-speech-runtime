use super::FailureReason;
use crate::{
    metrics::{
        LatencyHistogram,
        profile::{PacketTimings, RETAINED_TRACES},
    },
    protocol::{PacketSequence, SessionId},
};
use serde::{Deserialize, Serialize};
use std::{cmp::Reverse, time::Duration};
use tokio::sync::mpsc;

#[derive(Debug, Serialize, Deserialize)]
pub enum ClientPacketTrace {
    Echo {
        session_id: SessionId,
        sequence: PacketSequence,
        round_trip: Duration,
        client_start_delay: Duration,
        outside_server: Duration,
        server: PacketTimings,
    },
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
            Self::Echo { round_trip, .. } | Self::Failure { round_trip, .. } => *round_trip,
        }
    }
}

#[derive(Default)]
pub(super) struct PacketMeasurements {
    pub round_trip: LatencyHistogram,
    pub failed_round_trip: LatencyHistogram,
    pub client_start_delay: LatencyHistogram,
    pub outside_server: LatencyHistogram,
    pub slowest_packets: Vec<ClientPacketTrace>,
}
impl PacketMeasurements {
    pub(super) async fn collect(mut self, mut events: mpsc::Receiver<ClientPacketTrace>) -> Self {
        while let Some(packet) = events.recv().await {
            match &packet {
                ClientPacketTrace::Echo {
                    round_trip,
                    client_start_delay,
                    outside_server,
                    ..
                } => {
                    self.round_trip.record(*round_trip);
                    self.client_start_delay.record(*client_start_delay);
                    self.outside_server.record(*outside_server);
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
