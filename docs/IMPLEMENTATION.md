# Implementation record

The agreed target and acceptance checklist are in PLAN.md. This file records implementation choices, assumptions, review findings and validation evidence for the turn-based pipeline. Existing PERFORMANCE.md measurements concern the prior periodic echo workload.

## Decisions during implementation

- The initial worker transport is length-prefixed TCP, with strict JSON metadata and a separate binary PCM body. The public transport is WebSocket with strict control variants and binary audio frames.
- Interruption rejects subsequent backend token proposals. Previously accepted text remains in the ordered event stream and session record, even if socket delivery lags interruption. The archive records server acceptance; it does not imply client acknowledgement.
- Initial wire audio is mono PCM16 at 16 kHz. Other input formats and codecs are deferred to avoid changing the trained preprocessing implicitly.
- Generated-token accounting includes every server-accepted model token, including terminal EOS; a text delta may be empty because model tokens are not necessarily complete UTF-8 characters. EOS is recorded consistently rather than counted only when it carries a pending text flush. It is fed into the cache only after acceptance, before the next turn.
- The initial decoding policy is explicit greedy decoding. The model worker groups compatible prefills by appended prompt length to avoid padding contamination of recurrent state, and joins/splits attention plus convolution/recurrent state for dynamic decode batches. Cache joining currently copies tensors; real GPU profiling must determine its cost before claiming capacity.
- Cancellation before a queued prefill executes removes that work from model history; captured inputs remain in the audit record. A prefill already running may complete and extend the valid conversation prefix even after cancellation, while its unaccepted output proposal is discarded. Audit records preserve received inputs and accepted outputs; they are not yet a cache-recovery/replay API.

## Independent review

The Python review verified hybrid cache isolation and continuation and identified two concrete issues: admission omitted earlier unmaterialized cache reservations, and invalid byte-token UTF-8 could terminate a batch. Commit `c47d55c` fixes both, including padded workspace accounting and deterministic incremental replacement decoding. The post-fix Python suite reports 46 CPU tests passing and two hardware integration tests skipped.

The Rust review identified prefill starvation, a lost reservation when acceptance output is saturated, archive loss when close encounters a saturated worker mailbox, generation failures counted as successful benchmark turns, and cancellation delayed during opening. These fixes and final integration validation are in progress; their final evidence will be recorded below and in docs/LOCAL_VALIDATION.md.

## Pending hardware validation

Real checkpoint loading, GPU cache continuation parity, CUDA timing, peak VRAM, speech correctness, batching efficiency and concurrency at four generated tokens/second/session require the trained model and GPU node. No local mock result establishes those properties.
