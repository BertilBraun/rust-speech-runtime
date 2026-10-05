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

`Node::start` is async: each worker calibrates its device before admission opens. Capacity uses rolling p95 modeled device duration, a safety factor, the packet deadline and minimum arrival interval, compute headroom, a batch-fill reserve, and cache slots. Host observation delays have a separate bounded rolling p95; sustained delay beyond the scheduling margin pauses new admission, as does current queue pressure. An isolated host spike does not become fictitious GPU compute or trigger mass capacity shedding. If measured device capacity falls below existing load, excess sessions are explicitly terminated. A hard session cap is only an upper bound.

The default contract is one packet every 48 ms or more, with a 50 ms per-packet deadline. The simulator generates 1,600-byte PCM-sized packets (50 ms of mono 16-bit audio at 16 kHz). A shared pacing thread generates intervals independently of response completion; each session has a bounded one-packet input buffer. Each capture schedules the next 48–55 ms later, so a delayed timer never emits a catch-up burst. Actual capture intervals and timer delays are measured, including OS-induced gaps. Falling behind explicitly fails the session. An idle device starts ready work immediately by default; while it is occupied, the scheduler collects and prepares its successor batch. Optional idle collection is controlled by `--batch-wait-ms`. Deadlines belong to packets, rather than advancing on a session clock. One scheduler exclusively owns each worker's state.

The simulator waits until every session task has initialized its input loop before starting audio capture. Admission and setup durations are reported separately. Packet deadlines begin at capture; this readiness barrier keeps benchmark initialization outside the audio stream while retaining real pacing, socket and scheduler delays.

Each simulated device runs on a dedicated blocking-pool thread. Every nonempty ordinary batch has the configured compute duration, independent of how many of its slots are filled. Device completion on the simulated timeline is separate from when the host thread observes that completion. Host wakeup delay remains part of packet latency and deadline checks. Reports distinguish modeled device utilization from observed worker occupancy. Tokio remains responsible for ingress, timers and scheduling. There is one device thread per GPU, never one OS thread per session.

The default device wait is an OS sleep. Diagnostic alternatives are `--device-wait hybrid:200us` (sleep followed by at most a 200 microsecond spin) and `--device-wait poll:500ns` (repeated short sleeps until the absolute completion time). The requested polling interval is not a guarantee that the OS wakes the thread every 500 ns. Compare host completion delay and CPU usage before selecting a wait strategy.

Reports measure process CPU seconds and average logical cores for both the gateway and external simulator, plus CPU usage for each device thread. Process measurements cover serving/simulation through shutdown; device-thread measurements also include calibration and idle time. CPU usage is distinct from simulated device utilization: a sleeping mock device can be 100% occupied while consuming very little CPU.

The scheduler maintains an EDF selection while device work is running. Full batches can be submitted immediately; partial successors are submitted shortly before the preceding device completion. The bounded device command queue defaults to two waiting jobs in addition to one running job. Result handling does not gate the start of already-submitted work. Modeled batches execute serially, each for its full configured duration, on an autonomous device timeline; the host still waits for and observes every result. A host stall cannot retroactively delay an already-enqueued GPU kernel in this model, but it can still make its network result late. Closed sessions carry cancellation timestamps: work cancelled before its modeled start skips compute, while cancellation after start only invalidates its result. Queues and retained traces remain bounded.

`--device-queue-capacity 2` and `--launch-ahead-us 2000` expose device pipelining controls. Profiles report actual EDF preparation and job assembly CPU time separately from `scheduler_queue` residence before submission and `device_queue` residence after submission. The latter queue delays are not CPU time spent constructing a batch. Histograms preserve nanosecond samples and report milliseconds. Reports also count preparation updates during outstanding inference, batches submitted before their predecessor completes and the maximum number of outstanding device jobs. `--latency-safety-factor` exposes the admission service-time reserve for capacity experiments.

Audio packets carry a sequence and the fingerprint of their preceding prefix. Each worker owns a bounded cache pool containing fake KV-prefix fingerprints. Inference updates the entire logical prefix and returns the original audio bytes. Cache eviction preserves logical session history but invalidates GPU-local KV state. A subsequent cache miss requests replay of all preceding audio; replay is verified and contributes proportional simulated compute cost.

Closing a session releases its admission slot immediately, but a cache slot referenced by submitted device work remains retired until that work completes. A replacement can be rejected temporarily when all cache slots are still in use. Reports expose peak retired cache slots, and a test verifies that cancelled inference cannot reuse a replacement's cache memory.

A session retains at most one pending or running packet. Audio is never coalesced or silently replaced. A sequencing error, frame overload, prefix limit, impossible replay deadline or late completion explicitly fails the session. Cache misses are retryable without advancing the prefix. The external client retains the original audio history and automatically replays it after a cache miss. Replay shares the original packet's deadline. Closing/recreating a session assigns a new generation; old physical results cannot reach its replacement.

Admission protects against predictable overload, not arbitrary OS stalls or unforeseen hardware failures. Physical deadline overruns and terminated sessions remain visible in metrics. A late completion is reported as a failure, never as successful realtime audio.

Defaults admit at most 48 sessions per worker, or 384 across eight workers. Admission reserves a 5 ms scheduling margin and 10% of the remaining compute window: `(48 - 5) * 0.9 = 38.7 ms`, allowing three whole 12 ms batches, each with 16 slots. A simultaneous 48-packet burst therefore needs 36 ms of device compute; a fourth batch does not fit the admission budget. The mock kernel has a known fixed duration, so the default service safety factor is 1 and there is no additional partial-batch reserve. Both reserves remain configurable for experiments with uncertain device costs. Cache capacity and current queue/host pressure can lower admission. This is a model-based limit with host slack, not a hard realtime guarantee against arbitrary OS stalls.

Reports include client admission rejections, failed sessions by reason, successful and failed round-trip latency distributions, offered/echoed frames, pacing delay and cache replay work. Gateway metrics include queue delay, inference latency, scheduling latency, physical deadline misses, stale results, capacity terminations, channel saturation, batch fill and worker utilization. Every latency distribution includes p50/p95/p99/max. Read failures and terminations alongside latency: zero physical deadline misses alone does not prove healthy service. Client round-trip latency includes both network directions; the gateway's latency starts at ingress. The client checks the original capture deadline before accepting an echo.

Latency profiles separate ingress, worker mailbox, prefix validation, waiting before submission, waiting in the device queue, modeled device execution, host completion delay, result delivery and gateway return. Host device-thread wakeup delay is a separate diagnostic that can overlap modeled execution and is not added again to packet latency. Audio replies carry duration measurements, allowing the external client to measure capture-to-task delay and the remaining round-trip time outside the instrumented server. That remainder includes socket transfer, encoding/decoding and client echo verification; it is not a measurement of network transit alone. Cache replay retries also contribute to the remainder. Percentiles from different stages cannot be added to obtain a round-trip percentile.

Each worker and the client retain only their eight slowest packet traces. Server and simulator runtime monitors measure lateness of a five-millisecond timer without busy waiting or catch-up bursts. Device profiles separately measure `host_completion_delay`, the time between modeled completion and its observation by the device thread. This can include delayed wakeups and result-channel backpressure, so it does not identify a specific OS cause. Profiles have fixed-size histograms and bounded trace storage, and add clock reads, histogram updates and a small diagnostic payload to each echo.

Try `--phase aligned --min-interval-ms 50 --max-interval-ms 50`, `--evict-every 10`, or `--churn-secs 10`. Worker stress controls include `--slowdown-after-secs 3 --slowdown-ms 30` and `--worker-channel-capacity 1 --worker-input-delay-ms 20`. `RuntimeConfig`, `GatewayConfig` and `SimulationConfig` are the canonical configuration types; the CLI exposes common overrides. `examples/session.rs` demonstrates the direct library API.

Tests cover calibration/admission, EDF batching, sticky echoing, prefix continuity, cache miss/replay and replay cost, prefix limits, cancellation during inference, generation reuse, bounded mailbox overload, worker slowdown, timeouts and 2,000 concurrent sessions on eight device workers. TCP tests verify exact payloads, reclaimed admission slots, prefix replay, disconnects during physical inference, oversized frames, churn, and cancellation-safe message reception.

On this Windows machine, the 2026-10-05 release benchmark with eight 12 ms mock devices, 400 attempted sessions and 60 seconds of 48–55 ms packet arrivals admitted 64 sessions and rejected 336 before any audio processing. All 73,965 attempted packets were echoed: no failed sessions, deadline misses, capacity terminations or channel saturation. Round-trip p50/p95/p99/max were 22.4/25.6/26.2/39.1 ms. Batch fill was 16.3%, device utilization 72.8%, and node throughput about 1,221 packets/second. These are measured prototype results, rather than a universal capacity claim.

The three-second-per-case suite also completed low/50%/80%/95% load, capacity overload, aligned/random arrivals, jitter, churn and forced cache replay without session failures or physical deadline misses. Intentional worker slowdown and channel saturation explicitly failed affected sessions and exposed their causes in metrics. Local reports are in `benchmark-results/network-verified-400.json` and `benchmark-results/network-suite-final.json`. All 31 tests, strict Clippy and formatting checks passed. Improving the low batch-fill ratio while preserving the observed latency is the next performance experiment.

## Latency profiling, 2026-10-06

Three sequential release runs used eight workers, batches of up to 16, requested 12 ms inference, a 50 ms capture-to-echo deadline, 1,600-byte packets, random phases and 48–55 ms arrivals. Each attempted 400 sessions and ran for 60 seconds. These diagnostic runs raised admission headroom and batch-fill reserve to 1; the normal admission defaults were not raised.

| Per-worker session cap | Batch wait | Admitted | Failed sessions | Successful RTT p99 | Mean batch size | Device busy fraction |
| --- | --- | --- | --- | --- | --- | --- |
| 24 | 10 ms | 192 | 0 | 25.4 ms | 6.13 / 16 | 92.0% |
| 32 | 10 ms | 256 | 0 | 25.4 ms | 7.96 / 16 | 95.4% |
| 32 | 20 ms | 256 | 38 | 33.9 ms | 12.60 / 16 | 56.1% |

The last row includes time after 38 sessions failed, so its utilization is not a measurement at a sustained 256 active sessions. The zero-failure rows are observations from individual runs; earlier minute-long runs at 128 and 192 sessions failed, and none of these establish a reliable admission ceiling. Changes in host scheduling conditions and profiling instrumentation can affect timing.

At 256 sessions with a 10 ms batch window, mean ingress delay was 0.007 ms, worker mailbox and prefix validation each took about 0.004 ms, batch waiting about 6.7 ms, device dispatch about 0.05 ms, and result delivery about 0.035 ms. Mean physical inference was 12.34 ms per batch. The client measured 0.052 ms mean capture-to-task delay and 0.163 ms mean remaining round-trip time outside the instrumented server, including serialization, both socket directions and client processing. A two-second CPU sample during this run measured approximately one logical core combined for the gateway and simulator on this eight-logical-processor machine. Normal CPU/routing work was not the dominant latency source.

The tail traces expose a different problem. A 43.3 ms successful packet spent 18.8 ms waiting for a batch, 12.1 ms waiting for the device thread to begin, and 12.1 ms executing. In the 20 ms batch-window run, session 282 packet 675 spent 21.072 ms waiting, 0.006 ms in device dispatch, **37.879 ms inside the requested 12 ms blocking sleep**, and 0.518 ms in result delivery. Ingress, mailbox and prefix validation together took only 0.013 ms. Its client received a deadline rejection after 61.03 ms. The run recorded 25 server completion overruns, four stale results and 38 failed client sessions; the pacer also measured a 47.5 ms wakeup delay.

The measured failure path is delayed blocking-sleep completion plus existing batch wait. A device-dispatch interval contains channel handoff and thread scheduling, so a long dispatch does not by itself identify a Windows scheduler or Tokio defect. The CPU work inside routing is small, and the mock execution interval contains only `std::thread::sleep`; further system-level tracing would be needed to attribute that sleep overshoot to a particular OS mechanism. Increasing batch wait improves throughput efficiency but consumes deadline slack and did not meet the zero-failure requirement in this experiment. The next scheduling change should explicitly balance batch fill against the measured wakeup tail, rather than assuming GPU utilization alone predicts deadline safety.

Reproduce the comparisons with:

```powershell
cargo run --release -- benchmark --sessions 400 --duration-secs 60 --admission-headroom 1 --batch-fill-reserve 1 --max-sessions-per-worker 24 --output benchmark-results/profile-192.json
cargo run --release -- benchmark --sessions 400 --duration-secs 60 --admission-headroom 1 --batch-fill-reserve 1 --max-sessions-per-worker 32 --output benchmark-results/profile-256.json
cargo run --release -- benchmark --sessions 400 --duration-secs 60 --admission-headroom 1 --batch-fill-reserve 1 --max-sessions-per-worker 32 --batch-wait-ms 20 --output benchmark-results/profile-256-wait20.json
```

The profiles and bounded slow traces are included in each JSON report. Profiling validation covers injected ingress/mailbox/batching delay, complete stage accounting, TCP timing bounds and retention of the eight worst traces. All 33 tests, strict Clippy and formatting checks passed.
