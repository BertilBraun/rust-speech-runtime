# RTX 3090 decoder profiling and cache-copy optimization

Measured on 7 October 2026, using the same RTX 3090, checkpoint-9550 projector,
BF16 model weights and pinned PyTorch 2.6.0/CUDA 12.4 environment as the
[immediate-commit campaign](GPU_IMMEDIATE_COMMIT_3090.md).

## What the profile found

A short PyTorch CPU/CUDA operator trace exposed redundant cache initialization.
Our join/split code first concatenated or cloned tensors, then passed them into
Transformers' cache update methods. Those methods allocated more storage,
zero-initialized recurrent state and copied the supplied tensors again.
Attention initialization also concatenated the supplied tensors with an empty cache.
Even attention padding of zero positions cloned its input.

The change stays inside [the cache boundary](../backend/src/voice_worker/cache.py).
Joined tensors and independently cloned session tensors are installed directly
into the pinned SDK cache layers, including their initialization metadata.
Actual model forwards continue using the SDK's normal updates. Attention padding
is performed only when a row needs it. Session isolation, provisional-cache
isolation and bounded workspace reservations remain intact; this is not a paged
cache, a shared-view cache or a new scheduler policy.

Across three batch-16 decode steps at a 256-token initial context:

| Operator / runtime call | Before | After |
| --- | ---: | ---: |
| `aten::copy_` | 6,297 | 3,273 |
| `aten::zeros_like` | 1,836 | 0 |
| `aten::cat` | 870 | 258 |
| `cudaLaunchKernel` | 5,994 | 3,546 |

The profiled model matrix-multiplication device time stayed approximately 17.7 ms
over those three steps. Cache-related operator and launch counts fell sharply.
The profiler itself slows execution: its elapsed ranges must not be substituted
for normal request latency, and summed CUDA operator durations are not SM utilization.
The unprofiled comparison below establishes the speed improvement.

## Direct backend measurements

Each case prefills a synthetic newline-token context, warms five decode steps,
then measures twenty steps with independent per-session caches. Initial contexts
are 64 or 256 tokens and grow during decoding. Timing includes token embeddings,
cache join, model forward, cache split and the synchronized scalar token return.
There is no Rust gateway, audio encoding, admission, network or end-of-turn wait.
These are model row-steps, including any EOS proposals, rather than accepted
conversation output or a speech-quality test.

Milliseconds below are **p50 / p95 / p99 / max**. With twenty samples, p99 is the
sample maximum; these short trials do not establish production tail bounds.

| Initial context | Batch | Before | After |
| ---: | ---: | --- | --- |
| 64 | 1 | 21.06 / 21.10 / 21.13 / 21.13 | 19.87 / 19.97 / 20.50 / 20.50 |
| 64 | 8 | 29.98 / 30.86 / 30.92 / 30.92 | 22.62 / 22.75 / 22.79 / 22.79 |
| 64 | 16 | 39.20 / 39.34 / 39.48 / 39.48 | 25.95 / 26.06 / 26.07 / 26.07 |
| 256 | 1 | 21.07 / 21.13 / 21.17 / 21.17 | 19.95 / 20.05 / 20.06 / 20.06 |
| 256 | 8 | 30.00 / 31.29 / 31.89 / 31.89 | 22.64 / 22.70 / 157.61 / 157.61 |
| 256 | 16 | 39.55 / 39.82 / 40.43 / 40.43 | 26.13 / 26.28 / 26.79 / 26.79 |

Batch-16 median latency fell **34%**, and measured direct row-step throughput
increased from **405 to 612 steps/s** at the 256-token initial context.
The batch-8/256 case retained a **157.6 ms outlier**: its twenty-step aggregate
rate improved only from 265 to 272 steps/s despite the better median. Its cause
was not traced because operator tracing was a separate batch-16 run.

A forty-step repeat on the final source confirmed the improvement without
reproducing that outlier. It does not establish why the first outlier occurred.

| Initial context | Batch | Final-source repeat, p50 / p95 / p99 / max, ms |
| ---: | ---: | --- |
| 64 | 1 | 19.97 / 20.10 / 20.65 / 20.65 |
| 64 | 8 | 22.33 / 22.52 / 23.08 / 23.08 |
| 64 | 16 | 25.60 / 25.75 / 26.08 / 26.08 |
| 256 | 1 | 19.71 / 21.34 / 21.40 / 21.40 |
| 256 | 8 | 22.34 / 22.40 / 23.01 / 23.01 |
| 256 | 16 | 25.79 / 26.01 / 26.49 / 26.49 |

## Paired WebSocket workload

Two fresh gateway lifetimes used the same Linux binary, configuration and
checkpoint. The first used the baseline Python cache boundary and the second the
optimized boundary. Each lifetime ran two-turn calibration at 1, 2 and 4 sessions,
then two-turn cases at 16 and 32 offered sessions. Input was the same real
5.94-second appointment recording, delivered in 100 ms packets over loopback
WebSocket, with a 500 ms randomized start spread (seed 7), 250 ms think time,
immediate commit and no speculative preparation. Admission costs were retained
within each lifetime; the admission policy itself was unchanged.

| Offered sessions | Boundary | Completed / refused turns | Aggregate tokens/s | TTFT p50 / p95 / p99 / max, ms |
| ---: | --- | --- | ---: | --- |
| 16 | Before | 16 / 8 | 86.2 | 104.8 / 146.9 / 146.9 / 146.9 |
| 16 | After | 24 / 4 | 139.3 | 125.8 / 207.7 / 218.0 / 218.0 |
| 32 | Before | 32 / 16 | 151.2 | 144.9 / 333.1 / 333.1 / 333.1 |
| 32 | After | 64 / 0 | 267.3 | 318.0 / 822.8 / 898.0 / 898.0 |

All offered sessions opened. Refused clients stop attempting subsequent turns;
completed plus refused turns can therefore be smaller than twice the offered
session count. Aggregate throughput includes capture and think time and is not
decoder-only throughput. Changed batching can also change BF16-generated response
lengths, so the admitted workloads are not equal amounts of model output.

| Offered sessions | Boundary | Token gap p50 / p95 / p99 / max, ms | Gaps over 250 ms |
| ---: | --- | --- | ---: |
| 16 | Before | 32.3 / 32.7 / 35.3 / 227.3 | 0 |
| 16 | After | 27.4 / 28.0 / 75.3 / 233.9 | 0 |
| 32 | Before | 42.4 / 43.9 / 126.5 / 399.6 | 6 |
| 32 | After | 58.1 / 59.5 / 197.1 / 934.4 | 50 |

At 32 offers the optimized worker completed all 64 turns, with zero backend or
connection failures and zero rolling-rate violations. All 32 sessions had a full
two-second generation window; the lowest measured rolling rate was **10 tokens/s**.
This establishes functionality at that offered load in this short trial, not a
sustainable capacity guarantee or an improved first-token latency bound.

The faster backend admitted more work, exposing prefill queueing. Across the
optimized gateway lifetime, commit-to-prefill queue p95 was **692 ms**, while
encoding/projecting and Qwen prefill stage p95s were **30.3 and 39.1 ms**. Decode
stage p95 was **27.5 ms**, with a **165.5 ms maximum**. These distributions cover
different populations and their percentiles must not be added or subtracted.
Both gateways saved all 55 session archives with zero archival failures and
drained to zero active sessions; bounded archive backpressure was observed.

Higher accepted concurrency and worse TTFT can coexist. A separate first-token
latency objective, prefill admission and batching policy would need validation
before promising fast starts at this higher load. No headroom or capacity limit
was loosened to obtain the result.

## Reproducing the decoder profile

On an idle CUDA node with the configured checkpoint and pinned models available:

```bash
uv run python tools/profile_decode.py \
  --config /absolute/path/to/worker.json \
  --output /absolute/path/to/profile-results \
  --label cache-copy-optimization --trace
```

The tool saves every unprofiled sample, percentile summaries, model identity,
allocator peaks, CPU/CUDA operator tables and a Chrome trace. Keep the output
outside the source tree. Run the same command on the baseline source for a paired
comparison; use the same environment, configuration and resource guard.

The existing [GPU cache tests](../backend/tests/test_gpu_integration.py) passed
after the change: native BF16 continuation/full replay references, ragged
batching versus serial execution across decode steps, and prepared versus normal
multi-turn prefill with committed-cache isolation. Those numerical checks are
required when changing this version-specific cache boundary or upgrading Transformers.

## Scope of the result

The profile establishes avoidable overhead in our PyTorch cache handling. It does
not establish the GPU's maximum supported session count. Model execution still
uses eager launches and copies for isolated dynamic batches. Prefill still shares
the single per-worker execution lane with decode; higher-load first-token queueing
and conservative admission remain separate concerns.

Validation passed `cargo test --locked` (67 tests and one doctest), the explicitly
enabled Python-process/WebSocket integration test, strict all-target Clippy,
`cargo fmt --all -- --check`, `uv run ruff format`, `uv run ruff check --fix`, and
`uv run pytest tests tools/test_guard_worker.py -q` (81 passed, three CUDA tests
skipped locally). All three real-model CUDA tests passed on the 3090 and passed
again after the final helper extraction. Experimental services were stopped
after collection; the final device had 32 MiB used, no compute processes and zero
reported utilization. The container memory-limit failure counter remained zero.
Across 280 two-second resource samples, maximum child RSS was **2,435 MiB**,
minimum free VRAM **18,886 MiB**, and minimum available RAM **38,995 MiB**.
These sampled observations are not exact allocation peaks.

Raw evidence is retained under
`benchmark-results/3090-decode-optimization-20261007/`: both decoder traces,
unprofiled samples, operator tables, both gateway lifetimes, client reports,
session archives, resource samples and GPU test logs. The baseline was the
`f852436` serving snapshot. The paired measurements used cache source SHA256
`bd711a666156963be29396771c8ccac8deb795ab6244dec1fad01b581e2a1d95`;
the final source extracts the identical padding expression into a helper and
makes installation helpers private. GPU cache tests were rerun on that final
source, SHA256
`ef506ab10e9bd26d0d13f9b3bebe9451d7f6f4177664beb410910abbfd687295`.

The collected evidence bundle is `decode-optimization-evidence.tar.gz` in that
directory, SHA256
`856c2e73a9a0daf0e74fb1ca24f581af41d752d435264fe8fd6ad92e2836cf21`.
