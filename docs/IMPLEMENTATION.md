# Implementation record

The agreed target and acceptance checklist are in PLAN.md. This file records implementation choices, assumptions, review findings and validation evidence for the turn-based pipeline. Existing PERFORMANCE.md measurements concern the prior periodic echo workload.

## Decisions during implementation

- The initial worker transport is length-prefixed TCP, with strict JSON metadata and a separate binary PCM body. The public transport is WebSocket with strict control variants and binary audio frames.
- Interruption rejects subsequent backend token proposals. Previously accepted text remains in the ordered event stream and session record, even if socket delivery lags interruption. The archive records server acceptance; it does not imply client acknowledgement.
- Initial wire audio is mono PCM16 at 16 kHz. Other input formats and codecs are deferred to avoid changing the trained preprocessing implicitly.
- Generated-token accounting includes every server-accepted model token, including terminal EOS; a text delta may be empty because model tokens are not necessarily complete UTF-8 characters. EOS is recorded consistently rather than counted only when it carries a pending text flush. It is fed into the cache only after acceptance, before the next turn.
- The initial decoding policy is explicit greedy decoding. The model worker groups compatible prefills by appended prompt length to avoid padding contamination of recurrent state, and joins/splits attention plus convolution/recurrent state for dynamic decode batches. Cache joining currently copies tensors; real GPU profiling must determine its cost before claiming capacity.

## Pending hardware validation

Real checkpoint loading, GPU cache continuation parity, CUDA timing, peak VRAM, speech correctness, batching efficiency and concurrency at four generated tokens/second/session require the trained model and GPU node. No local mock result establishes those properties.
