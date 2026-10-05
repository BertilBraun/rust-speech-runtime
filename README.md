# Realtime voice inference runtime

The runtime admits persistent sessions onto sticky GPU-like workers, batches sequenced audio, and echoes processed payloads. The public API is demonstrated in `examples/session.rs`.

```powershell
cargo run --example session
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

`Node::start` is async: each worker calibrates its device before admission opens. Capacity uses the measured rolling p99 service duration, a safety factor, the packet deadline and minimum arrival interval, compute headroom, a conservative batch-fill reserve, and cache slots. A hard session cap is only an upper bound. Current queue pressure can pause admission. If measured capacity falls below existing load, excess sessions are explicitly terminated.

The default contract is one packet every 48 ms or more, with a 50 ms per-packet deadline. Full batches start immediately; partial batches wait at most the configured batching interval or the earliest safe launch time. Deadlines belong to packets, rather than advancing on a session clock. One scheduler exclusively owns each worker's state.

Each simulated device runs on a dedicated blocking-pool thread and sleeps for its configured latency. Tokio remains responsible for ingress, timers and scheduling. There is one device thread per GPU, never one OS thread per session.

Audio packets carry a sequence and the fingerprint of their preceding prefix. Each worker owns a bounded cache pool containing fake KV-prefix fingerprints. Inference updates the entire logical prefix and returns the original audio bytes. Cache eviction preserves logical session history but invalidates GPU-local KV state. A subsequent cache miss requests replay of all preceding audio; replay is verified and contributes proportional simulated compute cost.

A session retains at most one pending or running packet. Audio is never coalesced or silently replaced. A sequencing error, frame overload, prefix limit, impossible replay deadline or late completion explicitly fails the session. Cache misses are retryable without advancing the prefix. Closing/recreating a session assigns a new generation; old physical results cannot reach its replacement.

Admission protects against predictable overload, not arbitrary OS stalls or unforeseen hardware failures. Physical deadline overruns and terminated sessions remain visible in metrics. A late completion is reported as a failure, never as successful realtime audio.

Tests cover calibration/admission, EDF batching, sticky echoing, prefix continuity, cache miss/replay and replay cost, prefix limits, cancellation during inference, generation reuse, bounded mailbox overload, worker slowdown, timeouts and 2,000 concurrent sessions on eight device workers.
