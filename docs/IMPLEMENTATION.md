# Implementation record

The latest [RTX 3090 validation](GPU_BENCHMARK_3090.md) adds the completed 10 Hz checkpoint, real-model cache tests, throughput measurements and archive audits. The local validation counts and deferred hardware work below describe the initial implementation stage.

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

The Rust review identified prefill starvation, a lost reservation when acceptance output is saturated, archive loss when close encounters a saturated worker mailbox, generation failures counted as successful benchmark turns, and cancellation delayed during opening. Commits `734d3d3` and `e85feeb` fix these issues. An independent follow-up reran seven focused regressions successfully. Commit `9cc4da3` also fixes cost attribution after an interrupted prefill and adds its regression.

Already admitted prefills have a configurable maximum queue wait (2000 ms initially), after which they run despite a possible token-gap violation. This prevents indefinite starvation by a long nonpreemptive forward and reports the tradeoff in metrics. Solo-turn admission evaluates decode capacity separately from TTFT: a 300 ms prefill does not permanently block new turns when decode is fast.

Release-CLI validation found an idle worker transport timeout not caught by the original tests. Commit `8f63afb` preserves idle persistent connections and cache ownership while retaining a single timeout for a partial header/metadata/body. The expanded Python suite passes 52 CPU tests, with two hardware tests skipped. `54ecefa` clarifies device-stage timing boundaries. Final full-pipeline measurements and their failures are recorded in docs/LOCAL_VALIDATION.md.

The aligned overload run also exposed archive queue loss during simultaneous session closes. Commit `392deae` replaces lossy enqueue with bounded teardown backpressure: a closing connection retains its record and connection permit while waiting for the archive writer. Pending teardown records therefore remain bounded by the gateway connection limit, in addition to the archive mailbox and one writer. Disk failures remain explicit. The same change completes WebSocket close after a rejected session open and routes fatal gateway accept/setup/join errors through session, archive and runtime cleanup before returning the original error. An independent review found no remaining blocking issue and passed both new regressions. Fatal listener-error cleanup was assessed from source rather than an injected OS listener failure.

A subsequent release audit exposed a WebSocket close race: generation completed but the server dropped its reader before consuming the peer's acknowledgement. Commit `e010a42` completes the close handshake with one bounded deadline. Independent review passed both close regressions and verified that session archiving, writer flushing and peer close occur in order.

Commit `055aa05` extracts session cleanup and archiving from the connection entry point. The final cross-language process audit also found that terminating a Windows `uv run` launcher could leave its Python fixture child alive. Fixture commit `0507388` supports natural exit after the owning gateway disconnects; Rust test commit `74a5553` directly owns the resolved Python interpreter and waits for that exit. These changes are test infrastructure only, and the successful rerun left no fixture processes or listener ports behind.

## Verified local result

The final Windows suite passes 47 Rust tests; the separately enabled Python-process/WebSocket test passes four sessions and eight turns with 104 accepted tokens. Python passes 54 CPU tests with two GPU tests skipped. Formatting, strict Clippy, Rustdoc, release build and Ruff gates pass. Ubuntu/WSL passed the earlier full Rust suite and the affected final transport and archive tests. See [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md) for exact commands, revisions, measurements and failures found during validation.

The final aligned-overload release audit saved all 48 admitted session archives, with 21 bounded archive backpressure events and no archive, connection or backend failures. All 6,176 accepted tokens matched client, runtime and archive counts. The deliberate 350 ms decode run exposed token-gap and rolling-rate violations and rejected the next turn after profiling. These results validate orchestration with fixed-cost synthetic workers; they do not establish GPU throughput.

## Pending hardware validation

Real checkpoint loading, GPU cache continuation parity, CUDA timing, peak VRAM, speech correctness, batching efficiency and concurrency at four generated tokens/second/session require the trained model and GPU node. No local mock result establishes those properties.
