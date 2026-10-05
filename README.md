# Realtime voice inference runtime

The TCP gateway admits persistent sessions onto sticky GPU-like workers, batches sequenced audio, and echoes the exact input bytes. An external simulator sends 50 ms audio packets every 48–55 ms. GPU inference and KV state are simulated; socket transfer, serialization, scheduling and round-trip measurements are real.

```powershell
cargo run --release -- benchmark --workers 8 --sessions 400 --batch-size 16 --inference-ms 12 --duration-secs 60 --output network-benchmark.json
cargo run --release -- suite --duration-secs 5 --output scenarios.json
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

`benchmark` starts the gateway in its process and launches the simulator as a separate executable. It uses loopback TCP rather than calling the runtime directly. `suite` measures capacity first, then exercises low/50%/80%/95% load, excess session attempts, aligned arrivals, random phases, 48–55 ms jitter, churn, cache replay, slowdown and channel saturation. Stress scenarios intentionally cause explicit failures; they are reported alongside healthy scenarios.

For an independently running gateway and simulator, use two terminals:

```powershell
# Terminal 1; Ctrl+C drains the runtime and writes server metrics.
cargo run --release -- serve --listen 127.0.0.1:9000 --output gateway.json

# Terminal 2; can also run on another machine using the gateway's address.
cargo run --release -- simulate --address 127.0.0.1:9000 --sessions 400 --duration-secs 60 --output client.json
```

Each TCP connection owns a generation-scoped session lease. Old connections cannot send into, evict the cache of, or close a recreated session with the same ID. The in-process control API can target either a current session ID or a specific lease; the network gateway always uses its lease. Messages use a four-byte big-endian length prefix and bincode 2 serialization, with a 32 MiB maximum frame. The negotiated admission response carries assignment, deadline and audio limits. Connections use TCP_NODELAY; connection count, channel capacities, packet size and prefix size are bounded. This is a trusted gateway/runtime interface, with authentication handled upstream as specified in the project scope.

`Node::start` is async: each worker calibrates its device before admission opens. Capacity uses the measured rolling p95 device duration, a safety factor, the packet deadline and minimum arrival interval, compute headroom, a conservative batch-fill reserve, and cache slots. Using a bounded rolling p95 plus a safety factor avoids interpreting one isolated OS scheduling spike as permanent GPU slowdown. Scheduler delays are measured separately and consume packet deadline budget. A hard session cap is only an upper bound. Current queue pressure can pause admission. If measured capacity falls below existing load, excess sessions are explicitly terminated.

The default contract is one packet every 48 ms or more, with a 50 ms per-packet deadline. The simulator generates 1,600-byte PCM-sized packets (50 ms of mono 16-bit audio at 16 kHz). A shared pacing thread generates intervals independently of response completion; each session has a bounded one-packet input buffer. Each capture schedules the next 48–55 ms later, so a delayed timer never emits a catch-up burst. Actual capture intervals and timer delays are measured, including OS-induced gaps. Falling behind explicitly fails the session. Full batches start immediately; partial batches wait at most 10 ms or the earliest safe launch time. Deadlines belong to packets, rather than advancing on a session clock. One scheduler exclusively owns each worker's state.

Each simulated device runs on a dedicated blocking-pool thread and sleeps for its configured latency. Tokio remains responsible for ingress, timers and scheduling. There is one device thread per GPU, never one OS thread per session.

Audio packets carry a sequence and the fingerprint of their preceding prefix. Each worker owns a bounded cache pool containing fake KV-prefix fingerprints. Inference updates the entire logical prefix and returns the original audio bytes. Cache eviction preserves logical session history but invalidates GPU-local KV state. A subsequent cache miss requests replay of all preceding audio; replay is verified and contributes proportional simulated compute cost.

A session retains at most one pending or running packet. Audio is never coalesced or silently replaced. A sequencing error, frame overload, prefix limit, impossible replay deadline or late completion explicitly fails the session. Cache misses are retryable without advancing the prefix. The external client retains the original audio history and automatically replays it after a cache miss. Replay shares the original packet's deadline. Closing/recreating a session assigns a new generation; old physical results cannot reach its replacement.

Admission protects against predictable overload, not arbitrary OS stalls or unforeseen hardware failures. Physical deadline overruns and terminated sessions remain visible in metrics. A late completion is reported as a failure, never as successful realtime audio.

Defaults reserve 30% of the compute window, apply a 1.5 safety factor to measured service latency for admission, and reserve capacity for partial batches. Packet launch decisions use measured device time plus a scheduling margin, without applying the admission reserve a second time. A batch also launches immediately when every assigned session is ready, since further waiting cannot increase its size. Consequently, the old theoretical 52-session worker limit is an upper bound, not an admission promise. Random arrivals and OS scheduling can require substantially lower limits. Increase admission only after validating the complete network workload; treating 400 attempted sessions as 400 guaranteed admissions recreates the earlier overload problem.

Reports include client admission rejections, failed sessions by reason, successful and failed round-trip latency distributions, offered/echoed frames, pacing delay and cache replay work. Gateway metrics include queue delay, inference latency, scheduling latency, physical deadline misses, stale results, capacity terminations, channel saturation, batch fill and worker utilization. Every latency distribution includes p50/p95/p99/max. Read failures and terminations alongside latency: zero physical deadline misses alone does not prove healthy service. Client round-trip latency includes both network directions; the gateway's latency starts at ingress. The client checks the original capture deadline before accepting an echo.

Latency profiles separate ingress, worker mailbox, prefix validation, waiting for a batch/device, device dispatch, physical device execution, result delivery and gateway return. Audio replies carry duration measurements, allowing the external client to measure capture-to-task delay and the remaining round-trip time outside the instrumented server. That remainder includes socket transfer, encoding/decoding and client echo verification; it is not a measurement of network transit alone. Cache replay retries also contribute to the remainder. Percentiles from different stages cannot be added to obtain a round-trip percentile.

Each worker and the client retain only their eight slowest packet traces. Server and simulator runtime monitors measure lateness of a five-millisecond timer without busy waiting or catch-up bursts. Device profiles separately measure how much a blocking sleep exceeds its requested duration. Profiles have fixed-size histograms and bounded trace storage, and add clock reads, histogram updates and a small diagnostic payload to each echo.

Try `--phase aligned --min-interval-ms 50 --max-interval-ms 50`, `--evict-every 10`, or `--churn-secs 10`. Worker stress controls include `--slowdown-after-secs 3 --slowdown-ms 30` and `--worker-channel-capacity 1 --worker-input-delay-ms 20`. `RuntimeConfig`, `GatewayConfig` and `SimulationConfig` are the canonical configuration types; the CLI exposes common overrides. `examples/session.rs` demonstrates the direct library API.

Tests cover calibration/admission, EDF batching, sticky echoing, prefix continuity, cache miss/replay and replay cost, prefix limits, cancellation during inference, generation reuse, bounded mailbox overload, worker slowdown, timeouts and 2,000 concurrent sessions on eight device workers. TCP tests verify exact payloads, reclaimed admission slots, prefix replay, disconnects during physical inference, oversized frames, churn, and cancellation-safe message reception.

On this Windows machine, the 2026-10-05 release benchmark with eight 12 ms mock devices, 400 attempted sessions and 60 seconds of 48–55 ms packet arrivals admitted 64 sessions and rejected 336 before any audio processing. All 73,965 attempted packets were echoed: no failed sessions, deadline misses, capacity terminations or channel saturation. Round-trip p50/p95/p99/max were 22.4/25.6/26.2/39.1 ms. Batch fill was 16.3%, device utilization 72.8%, and node throughput about 1,221 packets/second. These are measured prototype results, rather than a universal capacity claim.

The three-second-per-case suite also completed low/50%/80%/95% load, capacity overload, aligned/random arrivals, jitter, churn and forced cache replay without session failures or physical deadline misses. Intentional worker slowdown and channel saturation explicitly failed affected sessions and exposed their causes in metrics. Local reports are in `benchmark-results/network-verified-400.json` and `benchmark-results/network-suite-final.json`. All 31 tests, strict Clippy and formatting checks passed. Improving the low batch-fill ratio while preserving the observed latency is the next performance experiment.
