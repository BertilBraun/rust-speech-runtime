# Turn-based speech inference runtime

Agreed 6 October 2026. This plan supersedes the old periodic audio-echo runtime. Historical measurements in PERFORMANCE.md describe that old workload and cannot predict the new model's capacity.

## Intended pipeline

An ordinary WebSocket client opens a session by ID, begins a user turn, sends ordered binary audio packets, and commits the complete utterance. The Rust gateway validates and buffers bounded PCM audio. A session stays on one configured GPU worker for its lifetime. At commit, the Python worker encodes the complete utterance with Whisper, projects it into language-model embeddings, prefills the persistent conversation cache, and performs one autoregressive decode step at a time. Rust schedules dynamically selected batches and streams committed text deltas back to the client. No TTS, authentication, database, or frontend is included.

The initial deployment is two RTX 3090s, but worker count and endpoints are configuration. The real backend is PyTorch/Transformers, not vLLM. Rust owns session lifecycle, admission, placement, queueing, batching, output acceptance and routing. Python owns model weights, tensors, opaque session cache state and model-specific operations. One actor owns each mutable scheduling state; all realtime queues and buffers have explicit bounds.

## Model contract

The source handoff is `C:/Projects/Voice-Full-Duplex-Light/docs/rust_scheduler_model_handoff.md`. Initial model: Whisper Small plus FP32 LayerNorm/pool-five/MLP speech projector into the BF16 Qwen3.5-2B text backbone. Use pinned revisions and a required projector checkpoint manifest. Preserve trained prompt markers, EOS 248046, and complete-utterance encoding. Whisper is bidirectional: audio packets cannot be appended as immutable language-model embeddings during capture. Initial wire audio is mono PCM16 at 16 kHz, at most 30 seconds per turn; adding formats requires explicit preprocessing.

Updated 7 October: the final retraining target is **10 Hz**, superseding the research project's archived 2.5 Hz selection. The existing pool-five adapter already matches 10 Hz. Default clients send 100 ms packets (1,600 samples / 3,200 PCM bytes); the gateway permits variable boundaries and a shorter final packet. This capture cadence is separate from the four-text-token/second generation objective. Final checkpoint selection is pending; see [model integration status](docs/MODEL_INTEGRATION.md).

The target history policy deliberately differs from the old reference: retain interleaved speech embeddings and accepted assistant tokens across turns using the model's complete hybrid cache (attention KV plus convolution/recurrent state). Model quality with this history is an assumption owned by training; serving correctness and cache isolation are independently tested. Cache positions, pending accepted token, prompt/model versions and sampling state are retained. Retain original audio and committed token IDs/text in a bounded in-memory session record and archive it on close. Never silently truncate history.

## Interruption and ordering

New user input can interrupt generation. The session actor defines the acceptance order. Stop scheduling the old turn, retain output accepted before the interruption, and discard proposals that arrive afterward. At most one backend operation per session is outstanding. A decode forward consumes only the previous accepted token; its next-token proposal is not fed into the cache until Rust accepts it. Reconcile the final accepted token and trained message delimiters before prefilling the next turn. Do not attempt generic attention-only cache rollback on Qwen's recurrent state. Generation/turn IDs fence backend proposals and label outgoing events. Already committed text events remain ordered before the next turn's acceptance; network delivery may lag interruption. Session records contain server-accepted output, not a claim of client acknowledgement. Duplicate commits are idempotent; missing, duplicate or out-of-order chunks fail explicitly.

## Scheduling and admission

Sessions reserve bounded memory/cache capacity and remain sticky. Admission of a user turn also accounts for active generation and queued prefill work, before accepting its audio. Measure completed encoding, prefill and decode durations, batch shapes, context lengths and memory usage. Use conservative reactive estimates and headroom; no fixed real-forward latency and no linear assumption that half a batch takes half the time.

After the first token, each actively generating session targets at least four model tokens per second. A 250 ms next-token scheduling target and per-session rolling throughput expose starvation; this is a service objective, not a hard operating-system guarantee. TTFT is separately measured. Fair decode selection and bounded prefill work prevent capture/prefill bursts from starving established generators. Prepare eligible work while another forward runs. Do not spin or use sub-microsecond sleeps in Tokio tasks.

## Public and worker boundaries

The public transport is WebSocket: strict JSON control/event variants and binary audio packets. Benchmarks use this exact interface. The internal transport is bounded length-prefixed TCP with strict JSON metadata and binary audio bodies, one persistent connection per GPU worker. This works on Windows and Linux without LibTorch in Rust. Protocol schemas, limits and model readiness are explicit. Backend operations are open, complete-utterance prefill, accepted-token decode, and close; batched requests/results carry session ID, turn/generation and operation identifiers. Workers report model capabilities and timing/memory observations. Startup/warmup readiness is distinct from liveness.

## Implementation checklist

- [x] Canonical protocols/configuration and typed backend boundary.
- [x] Bounded Rust session/worker actors, placement/admission and fair reactive batching.
- [x] Persistent cache lifecycle, interruption fences and dynamic batch membership.
- [x] Clean WebSocket gateway, binary packet validation and slow-client isolation.
- [x] Persistent Python worker, injectable test backend and real model adapter.
- [x] Per-session bounded records, asynchronous archives and bounded teardown backpressure with visible disk failures.
- [x] Ordinary WebSocket workload runner, multi-turn/churn/interruption/overload scenarios.
- [x] TTFT, token gaps/rates, queue/stage latency distributions, utilization and rejection metrics.
- [x] Independent architectural and correctness reviews; fix reported issues.
- [x] Rust fmt, strict clippy, tests and documentation; Python ruff and pytest.
- [x] Linux deployment/runbook and tomorrow's GPU/cache-parity benchmark checklist.

## Assumptions and validation limits

Numerical limits, throughput rolling windows, sampling policy, conservative unknown-cost admission, and archive format are implementation defaults, configurable and documented as they are chosen. Greedy decoding is the initial correctness policy; model/prompt configuration is versioned. A deterministic backend belongs in test/benchmark fixtures, injected through the same worker protocol rather than hidden inside production model code. Local CPU tests can verify orchestration and protocol behavior, but cannot establish actual GPU batching speed, BF16 parity, speech quality or the four-token objective at a given concurrency. These require the trained checkpoint and rented node tomorrow.

Completed coherent changes are committed iteratively. Implementation workers own disjoint areas; reviewers inspect boundaries and cancellation, cache and boundedness invariants. Follow-up decisions and verified evidence are recorded in docs/IMPLEMENTATION.md.
