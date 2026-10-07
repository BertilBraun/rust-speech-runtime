use std::time::Duration;

use bytes::Bytes;
use voice_scheduler::{
    protocol::{SessionEvent, SessionId, TurnId},
    simulation::DEFAULT_AUDIO_PACKET_MS,
    transport::VoiceClient,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client =
        VoiceClient::connect("ws://127.0.0.1:8080/v1", Duration::from_secs(120)).await?;
    let worker = client.open(SessionId("example-session".into())).await?;
    println!("assigned GPU worker {worker}");
    for index in 1..=2 {
        let turn_id = TurnId(index);
        client.begin_turn(turn_id).await?;
        for chunk in 0..10 {
            tokio::time::sleep(Duration::from_millis(DEFAULT_AUDIO_PACKET_MS)).await;
            client
                .audio(turn_id, chunk, Bytes::from(vec![0; 3200]))
                .await?;
        }
        client.commit(turn_id, 10, 16_000).await?;
        loop {
            match client.next_event().await? {
                SessionEvent::TextDelta { text, .. } => print!("{text}"),
                SessionEvent::Finished { .. } => {
                    println!();
                    break;
                }
                SessionEvent::Failed { code, message, .. } => {
                    return Err(format!("{code:?}: {message}").into());
                }
                _ => {}
            }
        }
    }
    client.close().await?;
    Ok(())
}
