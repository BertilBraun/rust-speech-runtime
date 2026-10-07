# Persistent model worker protocol, version 1

Rust types in `src/protocol/backend.rs` and frozen Pydantic types in `backend/src/voice_worker/protocol.py` define the same serialization boundary. GPU count, worker endpoints, capacities, timing estimates and model limits are configuration. The public WebSocket protocol is a separate boundary and accepts ordinary application clients.

## Framing and readiness

Each GPU worker accepts one persistent TCP connection. Every frame starts with an unsigned 32-bit big-endian JSON metadata length, followed by that many UTF-8 JSON bytes. Metadata is bounded to 1 MiB. Request metadata declares `body_bytes`; exactly that many raw binary audio bytes follow. There is no base64 conversion. Responses and readiness have zero body bytes. Configure TCP_NODELAY on both ends.

The backend sends readiness only after loading and warming its model. Its readiness object is `{ "type": "ready", "protocol_version": 1, "body_bytes": 0, "model_id": "...", "max_context_tokens": 16384, "max_batch_size": 16, "max_audio_samples": 480000 }`. Values above illustrate the shape, not hardware capacity. Rust validates capabilities and applies the smaller configured/backend audio, context and batch limits. Connection startup and every complete request/response have finite configurable timeouts. An idle persistent worker connection preserves its caches; the partial-frame timeout starts with the first header byte and covers the remaining header, metadata and body together. Unknown JSON fields are rejected.

## Requests and results

A request contains `request_id`, `body_bytes` and a nonempty `operations` array. A batch has one operation type: open, prefill, prepare, activate, discard_prepared, decode or close. Every operation has a distinct `operation_id` and an opaque `session_id`. The backend identifier includes a Rust allocation epoch, so closing and reopening the same public session ID cannot alias an older cache.

| Operation type | Additional fields | Contract |
| --- | --- | --- |
| `open` | none | Allocate worker-local session/cache ownership; acknowledge before public admission succeeds. |
| `prefill` | `turn_id`, `generation`, `audio_offset`, `audio_bytes`, `accepted` | Encode a complete mono PCM16 utterance at 16 kHz and append the trained audio/prompt embeddings to the persistent conversation cache. |
| `prepare` | Same fields as `prefill` | Compute against an isolated full hybrid-cache branch; retain its first token privately, preserving canonical state. |
| `activate` | `turn_id`, `generation` | Promote a matching provisional branch and return its held token without another GPU forward. |
| `discard_prepared` | none | Release a provisional branch while retaining canonical history; idempotent. |
| `decode` | `turn_id`, `generation`, `accepted` | Consume exactly the previously Rust-accepted token and propose one next token. |
| `close` | none | Release the complete cache, including recurrent and convolution states. Close is idempotent. |

Prefill audio slices are even-sized, contiguous and cover the binary body exactly. `accepted` is either null for prefill or an object `{ "turn_id": 1, "index": 0, "token_id": 123 }`. Decode requires the object. The index is zero-based within that assistant turn and distinguishes repeated token IDs. Prefill can reconcile the final accepted token of the previous turn before adding the next user utterance. A turn ID is unique within the public session but need not be numerically increasing; generation epochs increase.

A response contains the matching `request_id`, `body_bytes: 0`, `results`, `timing` and `memory`. Results appear in the same order as request operations, including backend compatibility regrouping. Each result repeats `operation_id`, `session_id`, and optional `turn_id`/`generation`. Rust checks identities and fences every inference completion. Its tagged `outcome` is one of `opened`, `closed`, `discarded`, `token` or `failed`.

`token` carries `token_id`, `text_delta`, `eos` and actual `context_tokens`. The proposal has not yet been consumed into the cache. Rust accepts and records a proposal only while the corresponding generation remains active. Accepted EOS is a model token, produces a TextDelta even when text is empty, and contributes to model-token counts. Its pending accepted token is reconciled into the next prefill, which must add the assistant end delimiter exactly once. Text deltas are incremental detokenizer output, not necessarily individual characters or words.

`failed` carries a closed-set error code shared with the public Rust ErrorCode and a message. Open capacity failure rejects only that attempted session. Any prefill/decode failure makes that session terminal because the protocol cannot prove whether the pending accepted token was consumed. Rust stops scheduling it, closes its cache and retains its accepted record for archiving. Connection/protocol failure removes the worker from placement while its actor remains available for final record collection; it does not silently rebuild a potentially inconsistent hybrid cache.

Optional preparation failure preserves canonical state and allows ordinary prefill fallback. Activation requires a matching session, turn, generation and unchanged canonical cache identity; missing or mismatched candidates return `invalid_state`. Preparation reserves one extra full-context cache per provisional session. More audio, cancellation or a new turn fences preparation and releases the branch. A `prepare` token is not a public response: only matching final commit and successful activation permit Rust acceptance. See [the candidate lifecycle diagrams](ENDPOINTING_PREPARATION.md).

## Cancellation, ordering and ownership

Each worker actor exclusively owns its scheduling state. It receives bounded commands while a separate execution task waits for backend I/O. There is at most one batch in flight per GPU and one operation in flight per session. Scheduler state remains responsive during inference; eligible captures, prefills, cancellations and output acceptance are processed concurrently with the backend forward.

Rust defines the commit point: a proposal successfully enqueued on the bounded session event stream is accepted and immediately recorded. New user turns retain all previously accepted tokens, stop the old generation and fence subsequent proposals. Already accepted output events remain ordered before the new Accepted event. This is server acceptance, not client receipt acknowledgment. A running decode consumes only an already accepted token, so discarding its next proposal never requires cropping recurrent cache state. Stale completion still updates the observed context length because valid previous work may have physically advanced the cache.

There are no hidden backend generation loops. Python never feeds a proposed token back into its cache until a subsequent Rust operation acknowledges it. Close waits for bounded mailbox space and in-flight work, then returns the final canonical session record. It never drops a record merely because the ordinary command queue is saturated. Slow consumers stop only their own session; record retention and finite output queues prevent global head blocking.

## Scheduling and measurements

Rust selects decode batches by earliest next-token target, and dynamically selects current eligible sessions. Prefill and Prepare form separate homogeneous batches up to `max_prefill_batch_size` (default four). A nonpreemptive prefill is normally launched when conservative estimates fit the next decode budget. Already admitted work must also make progress: after configurable `max_prefill_wait_ms` (default 100 ms), the oldest eligible operation class runs even if its measured cost exceeds the next-token budget. Ready activation takes priority over subsequent forwards. A long nonpreemptive forward can then violate the 250 ms token target, and metrics expose that tradeoff. This bounds waiting but does not make an individual whole-utterance forward preemptible. Model-specific compatibility grouping may subdivide a logical batch; reported Rust batch fill describes logical membership, not proof of one CUDA kernel.

The token target defaults to four model tokens per second after first output, with TTFT reported separately. Admission reserves active turn slots before capture and uses measured decode capacity and configured headroom. While other sessions generate, it also reserves a conservative prefill budget. A solo turn can therefore remain admissible with slow prefill and fast decode: first-token preparation has no 250 ms deadline. Existing active-turn interruption reuses its reservation. Unknown timing starts conservatively. Costs retain the last 32 measurements per operation phase, exact logical batch size and logarithmic context bucket. Predictions use recent maxima plus margin; unknown context uses a conservative same-size fallback, while unknown batch shapes retain the configured initial estimate. No prediction assumes half a batch costs half the latency.

`timing` contains nonnegative completed-work milliseconds: `elapsed_ms`, `encode_ms`, `prefill_ms`, `decode_ms`. Backend CUDA events time completed device stages, including transfers and cache copies; total RPC time also includes CPU preprocessing and transport. Device elapsed time is not a measure of GPU SM occupancy. `memory` contains observed `allocated_bytes` and `reserved_bytes`. Runtime metrics report distributions, decode batch fill, per-worker busy duration/utilization, memory, admissions, rejections, stale proposals, token-target misses and saturation. Utilization covers full worker RPC occupancy.

These policies and CPU fixtures establish orchestration correctness. Hardware throughput, compatible tensor batching speed, memory headroom and model quality must be verified with the trained checkpoint on the rented GPU node.
