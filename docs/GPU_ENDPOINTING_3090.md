# Preparing responses during endpoint detection on the RTX 3090

The gateway can now prepare the first response token before final end-of-turn confirmation. It encodes a complete candidate utterance, prefills against a private conversation-cache branch, and holds the result. A matching commit promotes that branch without another GPU forward. Resumed speech invalidates it. The [visual preparation guide](ENDPOINTING_PREPARATION.md) shows data flow, computation and cache ownership.

This reduces latency **after commit** by spending useful compute time during endpoint detection. It does not remove the endpointing interval or make Whisper a streaming encoder. The caller supplies candidate and final endpoint decisions; no voice activity detector is implemented.

## Hardware and comparison

These measurements use the same single RTX 3090, completed 10 Hz step-9550 projector, pinned Qwen3.5-2B and Whisper Small described in [the previous hardware benchmark](GPU_BENCHMARK_3090.md). Model serving weights are BF16; required recurrent accumulation remains FP32. Training and evaluation were inactive before testing. Clients connect to the real gateway over loopback WebSocket on the node.

Every client sends the same 5.94-second appointment recording as PCM16/16 kHz packets at 100 ms intervals. Most cases use starts spread across 500 ms, seed seven, two turns per session and a 250 ms think interval. Both paths wait **200 ms after the last audio packet** before commit. The prepared path sends `prepare` immediately after that packet; it does not wait for a ready notification before starting the confirmation timer. Reports retain commit-to-first-token, last-audio-to-first-token and actual endpoint confirmation time separately.

Runtime limits are 64 resident sessions, 32 active turns, batch size 16, prefill batch size four, a 100 ms prefill-wait threshold, a four-token/second generation target and 0.8 admission headroom. The backend uses a 2,048-position context limit, a 4 GiB cache reservation budget, 2 GiB workspace reserve and a 0.75 allocator-memory fraction. A resource guard stops only the serving/test child if free VRAM or available RAM falls below 4 GiB, or child RSS exceeds 6 GiB.

Final startup warmup covers encoder and prefill batch sizes one through four, fresh caches, both completed/interrupted history prefixes, six-second audio, 64/256-position prompts and decode batches up to four. Limits are configurable and clamped to worker capabilities. It runs before the listening socket opens and increases startup time. It deliberately has finite bounds and does not cover every future batch/context combination. Final admission uses only measured batch sizes that fit the active session count. A slow historical batch-16 observation therefore cannot reject a singleton that will execute batch one.

## Initial comparison and issues retained in the evidence

Before the admission correction, repeated warmed singleton cases returned two complete turns each: ordinary post-commit p95 was **48.5–48.6 ms**, prepared p95 **1.1 ms**. Last-audio-to-first-token p95 was **249.7–250.6 ms** versus **202.2–203.0 ms**. Two eight-session preparation runs completed all 16 turns each without rejection, with post-commit p95 **35.2/46.7 ms**, p99/max **35.2/46.7 ms**, and last-audio p95 **237.1/247.8 ms**. Their ordinary counterparts had p95 **575.0/920.6 ms**; the first also rejected four turns. These are small, sequential trials with different batching histories, not a controlled capacity guarantee.

A 16-session prepared case completed 32 turns but had **717.8 ms p95**, showing that a 200 ms confirmation interval cannot hide every queue or first-use stall. Its repeat completed 22 turns, rejected ten, and had **65.3 ms p95** among accepted turns. Afterwards the old admission calculation rejected every offered turn, even at lower load, because it priced a singleton using a slow batch-16 observation. That bug was corrected with a regression test; rejected cases and both gateway lifetimes remain in the raw evidence.

None of these trials had a backend failure or a rolling two-second generation-rate violation. Individual token gaps nevertheless exceeded 250 ms under burst load. A passing rolling-rate objective does not imply uniformly low token-gap latency.

After the admission correction but before expanded prefill warmup, a fresh worker's eight-session prepared run still had **610.8 ms p95**. The same lifetime measured a **515.1 ms maximum prefill stage**; its subsequent 16-session prepared run completed all 32 turns without rejection at **72.6 ms p95**. Eight intentional interruptions and a later singleton recovery also succeeded. This exposed the startup warmup gap: the original warmup exercised only one prefill row and the interrupted-history prefix. Final warmup now exercises every small prefill batch size and both history-prefix forms.

## Fresh service with the completed warmup

The final serving source is `aa6a87f`; Rust behavior is unchanged from `e8a451b`. After readiness, one-, two- and four-session cases populated the scheduler's cost observations before the larger comparisons. Actual GPU warmup does not populate Rust's reactive admission model.

| Case | Completed / rejected turns | Commit-to-first-token p50 / p95 / p99 / max, ms | Last-audio-to-first-token p95, ms |
| --- | --- | --- | --- |
| Eight prepared, first run | 16 / 0 | 12.4 / 35.9 / 35.9 / 35.9 | 237.3 |
| Eight ordinary | 16 / 0 | 84.2 / 113.7 / 113.7 / 113.7 | 315.4 |
| Eight prepared, repeat | 16 / 0 | 13.6 / 42.3 / 42.3 / 42.3 | 242.4 |
| Sixteen prepared | 31 / 1 | 17.2 / 57.1 / 59.6 / 59.6 | 258.3 |
| Four clients, cancel after two tokens, two turns | 8 interrupted / 0 | 0.7 / 34.6 / 34.6 / 34.6 | 235.8 |
| Singleton after the higher load | 2 / 0 | 0.9 / 1.1 / 1.1 / 1.1 | 203.3 |

All final cases had zero failed sessions and zero rolling two-second generation-rate violations. The 16-session case had a maximum individual token gap of **289.8 ms**, so the target remains a measured objective rather than a hard per-token guarantee. Its one refusal is excluded from the latency distribution and included in the admission count. Cancellation preserved exactly two accepted tokens per interrupted turn; later turns and singleton recovery remained healthy.

The final gateway lifetime recorded **79 preparations and 79 activations**, zero preparation fallbacks, eight stale proposals discarded during interruption, zero backend failures and zero active sessions at shutdown. It saved all 52 archives without error. Device-stage p50/p95/p99/max were encode **8.7/24.9/30.5/30.5 ms**, prefill **32.6/47.7/48.4/48.4 ms**, and decode **28.1/39.2/39.7/176.5 ms**. The earlier 515 ms prefill spike did not recur in this lifetime.

There were 1,619 decode batches with mean logical size **5.97/16**, or **37.3% fill**. RPC busy time was **51.4 seconds across 171.6 seconds**, or 29.9% of service lifetime, including audio capture and pauses between cases; this is not GPU SM utilization. Response lengths varied with batching and BF16 execution. These short trials use repeated identical speech and cannot establish sustained capacity, general output equivalence or a universal latency bound.

## Cache correctness and validation

The final guarded GPU run passed **84 tests**, including three real-model tests, in 56.3 seconds. Preparation of a second conversation turn leaves every canonical attention, convolution and recurrent tensor bit-identical. Its activated first token and complete hybrid cache match ordinary prefill exactly, including reconciliation of the previous accepted token.

An initial run passed 80 tests and failed the older fixed cached-versus-replay BF16 tolerance: two logits differed by up to 0.3046875. The corrected test independently compares runtime cached execution with native Transformers cached execution, and runtime replay with native full replay, both **exactly**, then requires equal greedy cached/replay tokens. It records native numerical drift rather than treating an arbitrary absolute tolerance as a scheduler invariant. The final native cached/replay maximum logit difference was **0.28125**. This fixture establishes the serving paths' parity with native execution; it does not guarantee identical BF16 choices near ties on every input.

Local validation passed `cargo test --locked` (67 tests), the explicitly enabled Python-process-to-Rust-to-WebSocket preparation test, `cargo fmt --check`, and strict all-target Clippy. Python validation passed `uv run ruff format`, `uv run ruff check --fix`, and `uv run pytest tests tools/test_guard_worker.py -q` (81 passed, three CUDA tests skipped locally). Tests cover queued/running/ready preparation, resumed audio, cancellation, stale activation, duplicate controls, count validation, optional cache refusal, failure fallback, homogeneous batching and bounded warmup shapes.

An audit of all **220 archives** verified exact original PCM for **369 committed turns**, contiguous token indices and monotonic timings. The archives' **37,897 accepted tokens** and **16 interrupted turns** matched client totals exactly. Every archive used worker zero and the same model identity. All three gateway shutdown reports had zero archive failures.

Across **604 resource samples**, maximum child RSS was **2,457 MiB**, minimum free VRAM **19,294 MiB**, and minimum available RAM **39,008 MiB**. Sampling is not an exact allocation-peak measurement. The container memory-limit failure counter remained zero. After final testing, both services were stopped and the GPU reported **32 MiB used**, **24,222 MiB free**, no compute processes and zero utilization. Automatic service start/restart remains disabled.

Raw benchmark reports, all three gateway summaries, session archives, test logs/JUnit and resource observations were copied to `benchmark-results/3090-endpointing-20261007/`. The compressed evidence SHA256 is `1233b047729a05688d2f09f7e89664bee0e6588776a9e8ff11082b2d5c42653e`. Checkpoints and raw results remain ignored by Git. The previous benchmark's local evidence contains the selected checkpoint snapshot.

## Reproduce

Start the supervised worker, wait until port 9100 is listening after warmup, then start the gateway. Deployment configuration is `endpointing-results/runtime.json`; the worker still uses `checkpoint-9550/worker.json`.

```bash
cd /workspace/voice-scheduler-integration-20261007
supervisorctl start voice-scheduler-worker
ss -ltn 'sport = :9100'
supervisorctl start voice-scheduler-gateway
./voice-scheduler benchmark --url ws://127.0.0.1:18080/v1 \
  --sessions 8 --turns 2 --audio-file appointment.pcm \
  --endpointing-ms 200 --prepare-before-commit \
  --report endpointing-results/new-prepared.json
```

Run the same command without `--prepare-before-commit` for the ordinary path. Keep the same endpointing delay. Record rejected turns separately from latency among accepted turns. Preserve the gateway shutdown summary and archives; stop the gateway before its Python worker. The worker and gateway listen privately on 9100 and 18080; Jupyter on 8080 is untouched.

Remaining measurements are longer mixed conversations, distinct recordings, realistic endpoint decisions, external network latency, model-quality parity and actual multi-GPU execution. This experiment does not establish sustained maximum concurrency or voice-response latency; TTS remains outside the project.
