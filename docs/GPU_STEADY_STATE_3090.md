# Staggered conversation benchmark and Rust boundary timings

Measured on 7 October 2026, source commit `6f2fda1`. One RTX 3090 served
8 and then 32 persistent conversations through loopback WebSockets. Session
starts were independently sampled across ten seconds. A separate twenty-second
measurement interval began after every session opened. This replaces the older
500 ms arrival pattern for ordinary benchmarks.

The measured Rust scheduling and transport work is small compared with model
execution at these loads. Admission still refuses some new turns at 32 sessions;
these results do **not** establish a refusal-free capacity of 32 conversations.

## Workload and measurement contract

- Real 5.94-second PCM16 speech, mono 16 kHz, sent in 100 ms packets.
- Persistent multi-turn conversations; each repeats the recording and waits
  250 ms after generation before attempting another turn.
- Random session-start offsets in `[0, 10000]` ms, seed 7.
- Clients start their conversations during ramp-up. Measurement starts once all
  open attempts finish, rather than synchronizing their first utterances.
- Refused turn starts retain the connection and retry after 250 ms in this run.
  Refusal counts are **attempts**, including retries, not abandoned utterances.
- No endpointing delay, speculative preparation, external network or client VAD.
- At measurement end clients stop starting turns and finish their current turn.

The nested `steady_state` client report counts token arrivals within the fixed
interval. Token gaps require both arrivals inside it. TTFT samples and completed
turns belong to turns committed inside the interval, including responses that
finish during draining. Retaining late first tokens avoids censoring latency tails.
The outer client totals and gateway timing distributions include ramp-up and drain.

First-token latency begins at an **admitted** turn's commit. It excludes capture,
waiting for admission, and a real application's endpoint detection. The last-audio
metric additionally includes the client's immediate commit scheduling interval.
All offered sessions remained connected and produced text; refused turn starts
did not remove clients from the cohort.

## Fixed-window results

| Observation | 8 sessions | 32 sessions |
| --- | ---: | ---: |
| Actual ramp-up | 8.60 s | 9.91 s |
| Measurement interval | 20.00 s | 20.00 s |
| Sessions producing tokens | 8 / 8 | 32 / 32 |
| Tokens received in interval | 1,979 | 4,984 |
| Aggregate tokens/s | 98.95 | 249.20 |
| Completed turns committed in interval | 16 | 63 |
| Refused turn-start attempts in interval | 0 | 212 |
| Sessions with complete rolling-rate windows | 8 / 8 | 32 / 32 |
| Rolling-rate observations / violations | 149 / 0 | 540 / 0 |
| Lowest observed two-second generation rate | 38.5 tokens/s | 22.5 tokens/s |
| Token gaps above 250 ms | 0 | 10 |

| Client latency, milliseconds | Samples | p50 | p95 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 sessions: commit to first token | 16 | 54.111 | 109.247 | 109.247 | 109.247 |
| 8 sessions: last audio to first token | 16 | 55.199 | 110.335 | 110.335 | 110.335 |
| 8 sessions: token gap | 1,963 | 22.847 | 24.991 | 70.079 | 130.239 |
| 32 sessions: commit to first token | 63 | 112.447 | 203.391 | 243.967 | 243.967 |
| 32 sessions: last audio to first token | 63 | 113.535 | 204.415 | 244.991 | 244.991 |
| 32 sessions: token gap | 4,921 | 29.615 | 61.759 | 162.815 | 403.199 |

Four tokens/s is evaluated over complete two-second windows after the first
token, sampled every 100 ms including stalls. All sessions had eligible samples.
Zero rolling violations does not imply every token arrived within 250 ms.
First-token percentiles have few samples, particularly at eight sessions.

## Where the time goes

These elapsed boundary timings cover each fresh gateway's **whole lifetime**.
They are instrumented durations, not a sampled CPU profile. Clock resolution is
one microsecond in the reported histograms; `0.001 ms` is the minimum bucket.

| Boundary at 32 sessions, milliseconds | Samples | p50 | p95 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rust worker command handling | 5,631 | 0.001 | 0.014 | 0.052 | 0.188 |
| Rust batch selection/building | 1,003 | 0.003 | 0.013 | 0.150 | 0.459 |
| Rust completion processing | 1,003 | 0.006 | 0.017 | 0.019 | 0.028 |
| Completion task to actor handoff | 1,003 | 0.002 | 0.002 | 0.005 | 0.051 |
| Prepared batch to execution task handoff | 1,003 | 0.001 | 0.002 | 0.004 | 0.029 |
| Paired backend RPC overhead | 1,003 | 0.477 | 0.698 | 0.900 | 1.547 |
| WebSocket event serialization | 9,803 | 0.001 | 0.001 | 0.002 | 0.015 |
| WebSocket send/flush | 9,803 | 0.006 | 0.016 | 0.031 | 0.259 |
| Python decode stage | 907 | 24.415 | 28.223 | 30.655 | 164.607 |
| Python audio encode stage | 41 | 15.079 | 28.847 | 31.839 | 31.839 |
| Python prefill stage | 41 | 32.591 | 35.231 | 65.183 | 65.183 |
| Committed turn's scheduling queue delay | 80 | 17.327 | 113.087 | 145.023 | 145.023 |

At eight sessions, batch building p95 was 0.003 ms, completion processing
0.006 ms, WebSocket send/flush 0.014 ms, and paired backend overhead 0.581 ms.
Python decode p95 was 23.375 ms.

`backend_rpc_overhead` records each Rust round-trip duration minus the matching
Python engine duration before aggregating. It includes framing on both sides,
loopback TCP and Python executor handoff; it is **not Rust-only** overhead.
Socket writes include OS waits and client backpressure. Do not subtract separately
aggregated p95 values or add these p95 values to derive a response's latency.

The evidence supports small measured Rust work at this load, not zero overhead
throughout the codebase. Ingress parsing, mailbox queue wait, metrics recording,
archive work and process-wide CPU utilization are not independently profiled here.
Queue delay is waiting for admitted model work to run, rather than CPU time spent
building batches. PyTorch stage durations include their CPU orchestration and GPU
work. The [earlier CPU/CUDA profile](GPU_DECODE_OPTIMIZATION_3090.md) identifies
redundant cache copies and initialization that were removed before this campaign.

## Admission and whole-run accounting

| Whole gateway/client lifetime | 8 sessions | 32 sessions |
| --- | ---: | ---: |
| Client elapsed time | 36.70 s | 38.77 s |
| Admitted / completed turns | 25 / 25 | 80 / 80 |
| Refused turn-start attempts | 29 | 607 |
| Completed turns per conversation, min–max | 3–4 | 2–4 |
| Mean decode batch size / maximum 16 | 3.31 | 9.80 |
| Decode batch fill | 20.7% | 61.3% |
| Backend RPC busy time / gateway lifetime | 61.3% | 70.1% |
| Successful conversation archives | 8 | 32 |

There were no session-open refusals, failed sessions, backend failures, channel
saturation events or archive failures. All conversations released their state.
The busy fraction is backend round-trip occupancy, **not GPU SM utilization**.

Each case uses a fresh gateway, so Rust cost estimates begin with the configured
conservative 100 ms unknown-forward estimate. The eight-session refusals occurred
entirely during ramp-up. At 32 sessions, refusals continued into measurement.
This keeps admitted generations fast by delaying new capture reservations.
It does not establish that the admission policy uses all hardware capacity or
that all 32 users could begin speaking immediately. Admission tuning and the
user-visible wait to begin a turn remain separate questions.

The Python model was warmed once and stayed loaded between cases; closing the
first gateway freed its sessions. Both cases reused compiler/kernel caches.
These are single short trials on repeated speech, with context growth over a few
turns. They do not establish sustained capacity, varied-utterance behavior,
long-context throughput, external-network latency or model quality.

## Configuration, resources and evidence

One RTX 3090, PyTorch 2.6.0+cu124, Transformers 5.13.0, BF16 model weights and
projector, checkpoint `9550`, Qwen3.5-2B and Whisper Small. The model/checkpoint
identities and kernel dependencies match the [decoder optimization report](GPU_DECODE_OPTIMIZATION_3090.md).
The optimized cache source SHA-256 is
`ef506ab10e9bd26d0d13f9b3bebe9451d7f6f4177664beb410910abbfd687295`.

Runtime: one worker, resident/active-turn limits 128, decode batch limit 16,
prefill batch limit 4, maximum context 2,048, four-token target, admission
headroom 0.8, unknown forward estimate 100 ms, prefill maximum wait 100 ms.
Backend limits: 4 GiB cache budget, 2 GiB workspace reservation, allocator
fraction 0.75. The gateway had two Tokio worker threads and 192 connection permits.

The existing guard sampled resources every two seconds. Across 66 observations,
minimum free VRAM was 18,834 MiB, minimum host-available RAM 39,042 MiB, maximum
Python child RSS 2,376 MiB, and maximum container cgroup usage 39,966 MiB against
a 46,170 MiB limit. Host-available RAM and container usage are different views.
Sampled observations can miss brief peaks. No resource guard fired. Services
stopped after the campaign; final GPU memory use was 32 MiB with 0% utilization.

Run against an already warmed deployment, replacing the recording path:

```powershell
voice-scheduler benchmark --url ws://127.0.0.1:18083/v1 --sessions 32 --measurement-secs 20 --audio-file appointment.pcm --minimum-packet-ms 100 --maximum-packet-ms 100 --start-spread-ms 10000 --think-ms 250 --endpointing-ms 0 --seed 7 --report client.json
```

Raw client reports, gateway reports, per-session archives, configurations, logs,
resource observations and the operator script were copied off the ephemeral node
to ignored `benchmark-results/3090-steady-state-20261007/`.
Bundle `steady-state-evidence-6f2fda1.tar.gz`, SHA-256:
`a7847ba052373dec285ebf09552daf961525189df2a4d8a3c368862471bb7ce4`.
Linux binary SHA-256:
`9de2bccbe229666d22c4fc01dc9a374abdec67b31ff32f51e5ca53c6b6ba1c45`.

Local validation: 73 Rust tests plus one doctest passed, strict Clippy,
`cargo fmt --check`, Ruff formatting/checks, and the explicit cross-language
Python-worker/WebSocket integration test passed. GPU results here exercise real
model serving; the three native cache-parity tests passed in the preceding cache
optimization campaign. No physical multi-GPU test was performed.
