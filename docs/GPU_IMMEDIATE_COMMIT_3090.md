# Immediate-commit speech serving on one RTX 3090

On 7 October 2026, the ordinary serving path completed two offered-concurrency sweeps from **8 to 96 conversations**. Clients streamed real speech in 100 ms packets, then committed immediately. Preparation was disabled and no endpointing delay was added. At eight offered conversations, both runs completed all turns with **106–109 ms first-token p95**. One 32-conversation run completed all 64 turns at **215 aggregate model tokens/s**, but first-token p95 rose to **673 ms**. Later runs rejected more turns as conservative cost observations accumulated. This is a runtime validation and admission experiment, **not a sustainable-capacity claim**.

## Workload and deployment

- One RTX 3090, 24 GiB VRAM, driver 550.107.02. Training and evaluation were stopped before serving started.
- Serving source **`f852436`**, built in release mode on Linux. One persistent Python worker and one Rust gateway; two Tokio runtime threads and one model-owner executor thread.
- Completed **step-9,550, 10 Hz projector**; SHA256 `8a1d62c7a8a13e430705c828605805181784603aa6f3d266024152df266c64c7`. Whisper Small and Qwen3.5-2B revisions are pinned in the worker manifest. All serving weights use BF16; model-required recurrent state and accumulation retain FP32.
- The [training project](https://github.com/BertilBraun/speech-llm-projection) selects step 6,775 for quality. This campaign deliberately keeps step 9,550 for runtime comparison with earlier measurements. It does not evaluate speech understanding or response quality.
- Every turn sends the same **5.94-second appointment recording**, mono PCM16 at 16 kHz. Packets contain 100 ms of audio, with the final shorter tail preserved. Both turns retain the same conversation cache.
- One-, two- and four-session calibration cases precede the sweep. Each client attempts two turns, with 250 ms between responses. Start times are randomly spread over 500 ms. Sweep one uses seed seven; sweep two uses seed 19. Both share one worker/gateway lifetime and the same evolving cost observations.
- All clients use the public `/v1` WebSocket interface over **node loopback**. There is no external-network measurement. The client knows the end of its recording and commits immediately; it does not implement VAD or predict the endpoint early.

### Limits and resource protection

Resident-session and active-turn caps were both raised to **128**, above the highest offered count. Gateway connection capacity was 192; command/event mailboxes were 128. Dynamic decode batches were bounded at 16, prefill batches at four. The generation objective was four accepted model tokens/second, admission headroom 0.8, unknown-forward estimate 100 ms and maximum prefill-wait threshold 100 ms.

The backend retained a 2,048-position context limit, 256-token response limit through Rust, ten-second utterance limit through Rust, 2 MiB session-record budget, 4 GiB cache reservation budget, 2 GiB workspace reserve and a 0.75 allocator-memory fraction. Startup warmup exercised six-second audio, 64/256-position prompts, encoder/prefill batches one through four, completed/interrupted history prefixes and decode shapes up to 16. Warmup finishes before listening; Rust cost observations are populated by the calibration clients afterward.

A Linux resource guard sampled every two seconds and stopped only its owned serving child if free VRAM or available RAM fell below 4 GiB, or worker RSS exceeded 6 GiB. No guard limit was reached. Existing training environments and management services were left unchanged. Both experimental services used disabled automatic start/restart and private ports 9101/18081.

## First-token latency and admission

All offered sessions opened successfully. The rejections below are **active-turn refusals**, not failed connections or failed inference. A client exits after a refused turn, so it does not attempt its remaining turn. Completed plus rejected is therefore the actual number of attempted turns, not always twice the offered session count. Every admitted turn finished with EOS; no turn hit the configured token limit.

| Offered sessions | Run | Completed / rejected turns | Commit-to-first-token p50 / p95 / p99 / max, ms | Last-audio-to-first-token p95, ms |
| --- | --- | --- | --- | --- |
| 8 | 1 | 16 / 0 | 66.7 / 106.1 / 106.1 / 106.1 | 107.2 |
| 8 | 2 | 16 / 0 | 69.5 / 108.6 / 108.6 / 108.6 | 109.6 |
| 16 | 1 | 32 / 0 | 113.6 / 173.2 / 193.0 / 193.0 | 174.2 |
| 16 | 2 | 30 / 1 | 126.8 / 256.9 / 257.0 / 257.0 | 257.9 |
| 32 | 1 | 64 / 0 | 306.9 / 673.3 / 781.8 / 781.8 | 674.8 |
| 32 | 2 | 30 / 17 | 165.2 / 239.1 / 287.5 / 287.5 | 240.3 |
| 48 | 1 | 30 / 33 | 218.5 / 321.8 / 369.7 / 369.7 | 322.8 |
| 48 | 2 | 16 / 40 | 117.1 / 197.4 / 197.4 / 197.4 | 198.5 |
| 64 | 1 | 30 / 49 | 237.6 / 367.1 / 367.1 / 367.1 | 368.4 |
| 64 | 2 | 16 / 56 | 113.3 / 220.9 / 220.9 / 220.9 | 222.1 |
| 96 | 1 | 30 / 81 | 217.2 / 371.5 / 373.2 / 373.2 | 372.5 |
| 96 | 2 | 16 / 88 | 113.0 / 196.5 / 196.5 / 196.5 | 197.5 |

Calibration first-token p95 was 55.4 ms at one conversation, 97.7 ms at two and 90.3 ms at four; all 14 turns completed. The sample counts are small, so p95/p99 can coincide with the maximum. Latency distributions include only accepted responses. Lower latency after many refusals is not evidence that the larger offered workload was served successfully.

**How to read latency:** TTFT starts when the client sends final `commit` and ends at its first received token. It includes scheduling queue delay, CPU audio processing, GPU Whisper/projector execution, Qwen prefill, IPC and WebSocket delivery. Last-audio latency starts at the last packet write and is about a millisecond higher here. Neither includes the time a real VAD needs to decide that the user stopped speaking. There is no extra 200 ms wait after that decision.

```mermaid
sequenceDiagram
    participant C as Client
    participant R as Rust scheduler
    participant P as Python / GPU
    loop 100 ms capture packets
        C->>R: Audio accumulates in bounded RAM
    end
    C->>R: Final commit, no extra delay
    Note over C,P: TTFT measurement begins here
    R->>R: Wait for eligible prefill slot
    R->>P: Complete utterance + cached prior history
    P->>P: CPU mel features, GPU Whisper + projection + Qwen prefill
    P-->>R: First token proposal
    R-->>C: Accepted first token
    Note over C,P: TTFT ends; autoregressive decode continues
```

## Generation throughput and gaps

No session violated the sampled **four-token/second rolling objective**. The lowest observed two-second rate was seven tokens/second in the first 32-session run. In total, 169 sessions had at least one complete rolling measurement window; short generations without a full window provide no rolling-rate evidence. This does not mean every gap was below 250 ms: there were **107 larger gaps**, including an 896 ms maximum while prefills competed with decoding.

| Offered sessions | Run | Token-gap p95 / max, ms | Gaps >250 ms | Minimum observed rolling tokens/s | Aggregate wall tokens/s |
| --- | --- | --- | --- | --- | --- |
| 8 | 1 | 32.4 / 156.8 | 0 | 28.0 | 88.1 |
| 8 | 2 | 32.4 / 188.5 | 0 | 26.5 | 87.1 |
| 16 | 1 | 43.0 / 285.4 | 9 | 18.5 | 154.0 |
| 16 | 2 | 42.5 / 338.4 | 4 | 20.0 | 153.1 |
| 32 | 1 | 86.5 / 896.0 | 50 | 7.0 | 215.3 |
| 32 | 2 | 41.8 / 409.3 | 10 | 20.5 | 149.4 |
| 48 | 1 | 41.8 / 420.4 | 11 | 18.5 | 154.5 |
| 48 | 2 | 32.2 / 216.6 | 0 | 29.0 | 89.7 |
| 64 | 1 | 41.6 / 422.1 | 12 | 20.0 | 154.7 |
| 64 | 2 | 33.0 / 220.5 | 0 | 29.0 | 89.5 |
| 96 | 1 | 41.7 / 421.9 | 11 | 20.0 | 154.4 |
| 96 | 2 | 32.2 / 217.7 | 0 | 29.0 | 88.7 |

Aggregate wall throughput divides received model tokens by the whole client-run duration, including audio capture, thinking and rejected clients. It is not pure GPU decode throughput. Counts include accepted EOS and empty text deltas; model tokens are not necessarily words. Responses and lengths can vary with batching and BF16 execution.

### Why admission changed between runs

The configured 128-session caps did not cause these refusals. Every rejection reported `capacity_exceeded: worker cannot reserve another active turn at target token rate`. The [cost model](../src/scheduler/costs.rs) keeps up to 32 observations per phase/batch/context shape, estimates with a recent maximum plus 10%, and prices unseen context buckets conservatively. Admission reserves decode rounds plus a prefill allowance within 80% of the 250 ms token interval. Capturing turns also hold reservations.

The same actor retained its observations throughout this sequential campaign. After the busier workload recorded slower forwards, later admission became more restrictive. This demonstrates a **conservative, history-dependent policy**, not an independent measurement of each concurrency level or a physical 15-/8-session GPU ceiling. The first 32-session result shows more work can run; its long TTFT and token gaps also show that rolling throughput alone is an incomplete latency objective. The benchmark does not establish how much further admission could safely be relaxed. No admission bypass or scheduler tuning was introduced to improve the reported numbers.

## Where time went

The gateway lifetime includes calibration and every sweep case. Histograms below are cumulative; their quantiles cannot be added to reconstruct an individual response. Device stages count selected requests/groups, not one sample per client token.

| Observation | Samples | p50 / p95 / p99 / max, ms |
| --- | --- | --- |
| Worker commit-to-first-token | 340 | 152.1 / 456.4 / 672.8 / 781.3 |
| Commit-to-prefill queue delay | 340 | 51.0 / 349.2 / 564.7 / 673.3 |
| Complete backend RPC, including control requests | 5,050 | 31.9 / 43.0 / 105.5 / 185.3 |
| GPU encoding/projection stage | 149 | 15.1 / 30.2 / 30.3 / 30.3 |
| GPU prefill stage | 149 | 34.3 / 47.7 / 48.9 / 49.0 |
| GPU decode stage | 3,933 | 30.5 / 39.8 / 40.4 / 181.6 |
| Worker token gap | 38,325 | 41.0 / 85.1 / 155.1 / 896.0 |

Higher-load TTFT is substantially affected by waiting for prefill, rather than a 673 ms Whisper forward. CPU waveform/mel preparation is included in the RPC, not separately exposed in the encoding-stage metric. CUDA event durations include cache joins/splits and host launch gaps; they are not GPU SM busy time.

Decode batches averaged **9.74/16**, or **60.9% logical fill**. Backend request busy time was **146.6 seconds over 327.0 seconds**, or 44.8% of the gateway lifetime. That denominator includes capture and gaps between cases; it is not SM utilization or proof of unused sustained decode capacity.

## Whisper streaming limitation

Incoming 100 ms packets currently accumulate in Rust RAM. **No model prefill occurs per packet on the ordinary path.** A session retains its full prior-turn hybrid cache; only the current complete utterance needs encoding and prefill after commit.

Whisper uses bidirectional attention. Adding future audio can change earlier encoder outputs, so previously prefilling a partial utterance cannot simply be extended into the same final cache with equivalent semantics. The optional `prepare` path recomputes a complete candidate on a private cache branch and holds its first token until a matching commit. New audio invalidates that branch. Calling it continuously would repeatedly encode and prefill growing candidates and reserve extra cache; it would not become incremental streaming inference.

The [earlier preparation experiment](GPU_ENDPOINTING_3090.md) knew the recording's final packet and then inserted 200 ms of confirmation time. Its low post-commit latency measures optimistic overlap, not a realistic endpoint predictor. This campaign removes that interval completely. A real client sends final commit when its VAD decides; a compatible streaming encoder/model would be needed to move current-utterance prefill into packet arrival without repeated whole-candidate computation. Continuous speculative preparation was intentionally not added or benchmarked.

## Correctness, resources and evidence

The gateway served **535 opened conversations**, **340 completed turns**, and **38,665 accepted model tokens**, with **365 refused turns**. It reported zero failed connections, backend failures, stale results, channel saturation events and preparation operations. At shutdown no sessions remained active. All **535 archives** were saved with zero disk failures; ten connection teardowns waited on bounded archive backpressure.

The local archive audit verified byte-for-byte original PCM for all 340 committed turns, contiguous token indices and monotonic elapsed timings. Its 38,665 accepted tokens exactly matched client totals. Every archive retained worker zero and the same model identity, `f61f38fffaebfd5443be7bd3d0936d06c849b2c9089c88982ce031047ac6a1ab`. This campaign contains no intentional interruptions; the earlier hardware report records that separate validation.

Across 193 resource samples, maximum worker RSS was **2,369 MiB**, minimum free VRAM **19,044 MiB**, and minimum available RAM **39,071 MiB**. These are sampled observations, not exact allocation peaks. The container memory-limit failure counter remained zero. Both experimental services were stopped after the campaign; the final GPU reported 32 MiB used, 24,222 MiB free and zero utilization. Final device/process state is retained in provenance. No training process was interrupted.

Validation of the serving revision passed Linux `cargo test --locked` (67 tests and one doctest), local `cargo fmt --all -- --check`, strict all-target Clippy, `cargo doc --locked --no-deps`, and the explicitly enabled Python-process-to-WebSocket test. Python validation passed `uv run ruff format` (29 files unchanged), `uv run ruff check --fix`, and `uv run pytest tests tools/test_guard_worker.py -q` (81 passed, three CUDA tests skipped locally). The earlier hardware campaign's [84-test result](GPU_ENDPOINTING_3090.md#cache-correctness-and-validation) includes three real-model cache tests; those tests were not rerun in this final workload campaign.

Raw client reports, deployment configuration, logs, resource samples, provenance and session archives are preserved locally under `benchmark-results/3090-immediate-commit-20261007/`. The compressed evidence SHA256 is **`08ed0e9e0060d05ae2bef15e9a01e756880a45a7872da24bc98c172c77c9bd9a`**. Raw audio, weights and bulky evidence remain ignored by Git.

## Reproduce the ordinary path

Start a warmed backend and gateway using the [backend deployment guide](../backend/README.md) and [client/configuration guide](CLIENT_GUIDE.md). Use the same checkpoint/configuration, calibrate costs first and record offered/admitted work separately. This campaign's on-node client command was:

```bash
./immediate-voice-scheduler benchmark --url ws://127.0.0.1:18081/v1 \
  --sessions 32 --turns 2 --audio-file appointment.pcm \
  --minimum-packet-ms 100 --maximum-packet-ms 100 \
  --start-spread-ms 500 --think-ms 250 --endpointing-ms 0 --seed 7 \
  --report immediate-results/sweep-32-1.json
```

Run offered counts 8, 16, 32, 48, 64 and 96 after the one/two/four-session calibration. Repeat with seed 19 without resetting the gateway to reproduce this campaign's history dependence. Omit `--prepare-before-commit`. Stop the gateway gracefully before its backend to obtain the final metrics and drain archives.

Remaining work is validation with the selected quality checkpoint, longer varied conversations, real client endpoint decisions, external network latency and physical multi-GPU execution. This final campaign verifies the current ordinary pipeline and exposes admission/latency tradeoffs; it does not establish a maximum deployable conversation count or voice-output latency.
