# GPU validation checklist

The turn-based pipeline targets configurable PyTorch workers. Local orchestration tests and synthetic timing runs are not model performance measurements. Run this checklist after the trained checkpoint and rented GPU node are available.

A limited [shared RTX 3090 check](DEPLOYMENT_3090.md) now passes with an intermediate 10 Hz checkpoint. It covers one native-cache/replay case and small real-speech/cancellation runs. The checklist below remains necessary for final-checkpoint parity, broader batching and capacity validation.

## Record the deployment

Record GPU model/count, VRAM, driver, CUDA, PyTorch and Transformers versions, model/tokenizer revisions, projector SHA256, prompt version, decoding settings, scheduler limits and Git revision. Preserve the full configuration with each benchmark. Two 3090s are an initial experiment, not a topology invariant.

Start one persistent backend worker per selected device. Load model weights once, set evaluation/inference mode and finish warmup before marking ready. Verify that the Rust gateway refuses new sessions while all workers are unavailable. Expose only the WebSocket gateway to benchmark clients; the backend IPC ports are local/internal.

## Establish correctness before capacity

1. Compare complete audio preprocessing, Whisper retained frames and projected embeddings against the training reference. Include a partial pooling block, short audio and the 30-second limit.
2. Check prefix/suffix token IDs and the tokenizer EOS override (248046). Confirm that the current utterance enters as speech embeddings, without a transcript or training label.
3. Compare greedy single-turn logits/output within a documented numerical tolerance; investigate near-tie differences rather than hiding them.
4. Compare two-turn cached continuation with full replay of the same interleaved speech/text prefix. Attention KV and convolution/recurrent state must both be preserved.
5. Interleave two sessions with distinct audio and histories. Neither cache may affect the other. Repeat after close and session-ID reuse.
6. Interrupt before prefill completion, between decode steps and during an in-flight decode. Retain exactly accepted output, discard late proposals, reconcile the final accepted token and continue the next user turn without attention-only rollback.
7. Compare batch size one with supported multi-session batches, including differing context lengths, changing membership and EOS/interruptions within a batch.
8. Force configured context/cache limits and an allocation failure. Verify explicit error handling, complete resource cleanup and continued service for unaffected sessions.

The model's quality with persistent speech history is owned by training, but cache continuation correctness still requires these serving tests.

## Measure the complete service

Use ordinary network WebSocket clients with independent random session starts and 100 ms packets during capture, matching the final 10 Hz model target. Include the 100–110 ms jitter scenario, different utterance lengths, think times, output lengths, simultaneous end-of-turn bursts, session churn and interruptions. Preserve short final packets and exact sample counts. Do not force sessions into a periodic audio-decode workload.

Measure end-of-turn to first token separately from subsequent client-observed model token throughput. Report per-session rolling rates and inter-token p50/p95/p99/max; aggregate tokens/second alone can conceal starvation. For actively generating sessions, the initial service objective is at least four model tokens/second after the first token. Report violations and the measurement window explicitly.

For every load level, record admission/rejection, active sessions/turns, queue delays, encoding/projector/prefill/decode timings, batch shapes/fill, worker busy time, client latency, channel saturation, stale results, archive failures, history lengths and peak VRAM. Use completed CUDA work timing, not just asynchronous Python submission duration. CPU orchestration and network overhead are separate measurements; percentile components cannot be added to recover end-to-end percentiles.

Begin at low load, then increase until the first repeatable quality or memory limit. Tune conservative cost estimates, reserve/headroom and batch budgets against measured distributions. Repeat sustained runs at the proposed capacity and at overload. A short passing test does not establish a reliable admission limit. Also test a slow GPU and a slow WebSocket consumer: overload must stay bounded and visible without starving all other sessions.

Keep configuration, raw reports and selected session archives with the result. Document failures as well as passing runs. No capacity number from the previous mocked echo scheduler should be reused as an admission claim for this model.
