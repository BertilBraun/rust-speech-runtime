# Persistent PyTorch speech worker

One process keeps one GPU's Whisper encoder, FP32 speech projector, BF16 Qwen text model and session caches resident. Rust selects every batch and accepts every output token. The worker performs complete-utterance encoding/prefill or one decode step; it never starts an independent generation loop. The public application's audio and text remain in the Rust session record. Device count is determined by the number of configured worker endpoints.

## Local validation and explicit benchmark fixture

From this directory in PowerShell:

```powershell
uv sync --locked
uv run ruff format --check
uv run ruff check
uv run pytest -q
uv run python tests/fixture_worker.py --port 9100 --response-tokens 12 --prefill-ms 20 --decode-ms 12
```

The fixture is a separate test executable implementing the production TCP protocol. It returns labeled deterministic text, models bounded session/context capacity, and charges the same configured time for every nonempty batch up to its limit. It supplies no ML predictions or GPU performance evidence. Start separate instances on different ports when testing multiple worker endpoints. Its parameters include `--max-sessions`, `--max-batch-size`, `--prefill-ms`, `--decode-ms` and `--response-tokens`.

CPU tests run the actual Transformers 5.13 Qwen hybrid model with small random weights, not a mock cache. They compare attention, convolution and recurrent state and logits for ragged decode batches, chunked multimodal-style continuations, changing membership and interruption reconciliation. Test-only model and CUDA-memory fixtures exercise the production engine's dispatch and validation. GPU integration tests are marked and skipped unless `VOICE_WORKER_CONFIG` and a CUDA device are available.

Validated local lockfile: Python 3.12.13, PyTorch 2.14.1 CPU, Transformers 5.13.0, Pydantic 2.13.5. Python 3.10 or newer is supported by the project metadata; numerical checks were run on Python 3.12. PyTorch's CPU execution is used solely for these tests; the production worker requires CUDA.

## Linux GPU deployment

Use a CUDA-capable Linux node with sufficient driver support for the locked PyTorch wheel. The worker uses the node's existing driver; it does not install or change it. Initial deployment can use two 3090s, with one worker configuration and process per device.

1. Copy the repository and the trained `projector.safetensors` to the node. Copy `config.example.json` to a deployment configuration and update the checkpoint path and **actual SHA256**. The example identifies the handoff's 6 October checkpoint; a newly trained checkpoint needs a new manifest hash.
2. In `backend/`, create the environment using `uv sync --locked`. Check actual driver/runtime visibility before loading models:

   ```bash
   nvidia-smi
   uv run python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count())"
   sha256sum /absolute/path/to/projector.safetensors
   ```

3. Set `device` to `cuda:0`, `cuda:1`, etc. and assign distinct TCP ports. Alternatively, give each process one visible GPU with `CUDA_VISIBLE_DEVICES` and use `cuda:0` inside each process. Gateway worker endpoints must match these ports. Bind to loopback when gateway and workers share a node.
4. Start a process for each chosen configuration:

   ```bash
   uv run voice-model-worker --config /absolute/path/to/worker-0.json
   ```

   Keep worker processes under the deployment's process supervisor. Model snapshots download at their manifest-pinned revisions; a cached Hugging Face snapshot avoids repeated downloads. Startup verifies the checkpoint digest, loads the text-only Qwen class and only Whisper's encoder onto the GPU, sets evaluation/inference mode, verifies prompt token IDs and EOS, and warms complete audio prefill plus cached decode before the listening socket opens. Readiness is a framed capability message after a gateway connects.

5. Run hardware cache tests before accepting capacity claims:

   ```bash
   VOICE_WORKER_CONFIG=/absolute/path/to/worker-0.json uv run pytest -q -m integration
   ```

   Repeat for every configured GPU, including nonzero device indices. GPU CUDA events explicitly use the model's device stream. Compare the full speech route against saved reference audio/features and next-token logits from training, then run the public WebSocket workload runner. The included integration tests compare BF16 cached continuation/full replay and ragged batching/serial logits using complete synthetic audio; they do not establish speech quality or replace reference-checkpoint parity testing.

## Cache and batching invariants

Each session owns a Transformers `DynamicCache` configured for the pinned Qwen model. All full-attention K/V and linear-attention convolution/recurrent tensors are retained. No attention-only cropping, rollback or generic batch-selection method is used.

Decode batches can contain sessions with different context lengths. Joining left-pads **attention tensors only**, builds a matching attention mask, and supplies each row's true absolute positions. Fixed-size convolution/recurrent state is concatenated along the batch dimension. Splitting removes only temporary attention padding and clones every session's state so changing batches cannot alias another session's tensors.

Prefill first encodes audio as one Whisper batch, projects each real cropped sequence before padding, and groups language-model forwards by new prompt length and fresh/populated cache state. Each group executes a real batched forward. This avoids corrupting recurrent state by padding new tokens. Different speech lengths can therefore produce several forwards within one Rust-selected request; timings include all groups. There is no permanent batch membership, and no assumption of linear cost versus batch fill.

Rust sends `AcceptedToken {turn_id, index, token_id}` with decode and optionally with the next prefill. A forward consumes only that accepted token and proposes the next token. Proposal bytes and detokenizer state are committed only after matching Rust acceptance. Equal token IDs at successive positions remain distinct through the index. A new user turn reconciles a pending final accepted token exactly once, closes the assistant message once with EOS, inserts the newline, then adds the trained user prefix, speech embeddings and assistant suffix. An in-flight proposal rejected after interruption is never consumed. Generation epochs must increase; public turn identifiers need not be numerically ordered.

Byte-level detokenization buffers incomplete UTF-8 characters. EOS flushes any previously accepted incomplete bytes with replacement text. Cancelling or hitting a token limit can end before a complete UTF-8 character exists; exact accepted IDs remain recorded, while the incomplete character is not invented or sent. The trained suffix already opens the assistant role with an empty thinking block; there is no additional audio or BOS token.

EOS itself is an accepted, recorded model token, including when its text delta is empty. Generated model-token counts include EOS; the fixture's configured ordinary response tokens are followed by one EOS token. A later prefill consumes that pending accepted EOS without inserting a second assistant-end delimiter.

Session admission reserves the estimated maximum configured context's attention KV plus FP32 recurrent and BF16 convolution state against `cache_budget_bytes`. Actual free device memory, `max_sessions` and `workspace_reserve_bytes` also gate allocation. The reserve is conservative and fixed per admitted session; it is not a paged allocator or a measured throughput guarantee. Attention state grows only to `max_context_tokens`; context exhaustion is explicit and never silently truncates history. Closing a session or losing the gateway connection releases owned caches. A failed model forward invalidates affected sessions' caches and reports `backend_failed`.

## Measurement and tomorrow's assumptions

Forward timings use CUDA events on the device stream, including cache join/split. Returning scalar proposed token IDs already synchronizes the completed batch, so timing does not add another per-token device synchronization. Total elapsed time includes Python/CPU overhead. Encoding timing covers the encoder/projector stage; CPU feature extraction and host work are also included in total elapsed. Responses report actual CUDA allocated and allocator-reserved bytes.

The implementation currently copies hybrid tensors when joining and splitting dynamic batches. This provides simple ownership and isolation, but its allocation/bandwidth overhead must be profiled on the rented node. Left-padding very unequal contexts also adds attention work. Neither cost has a hardware performance claim. Equal-new-length prefill grouping may reduce fill for irregular audio durations; report both Rust-selected and actual model-group batch sizes when profiling.

Transformers supports a pure PyTorch fallback for Qwen's linear attention when optional fast kernels are absent. The lockfile intentionally uses that supported path initially; kernel optimization, CUDA graphs, paged caches and packed variable-length prefills require separate parity and speed validation. One-session warmup does not eliminate every later batch/context shape's first-use overhead.

The retained interleaved speech/text history is the agreed serving target and a training-owned model-quality assumption. The runtime supports it with persistent hybrid state. Tomorrow's measurements must establish real audio TTFT, token-gap p50/p95/p99, per-session rolling token rate, encoding/prefill/decode cost by batch/context, queue delay and peak VRAM, then tune scheduler headroom and capacity to the four-token-per-second objective. The old audio-echo benchmark is unrelated to this model's capacity.

The strict version-one metadata and PCM wire contract is documented in the repository's `docs/WORKER_PROTOCOL.md`; `protocol.py` mirrors the Rust types. Only one gateway connection owns a worker at a time. Frames, bodies, active caches and transport wait durations have bounds; model forwards run on one dedicated thread so Python's network event loop remains responsive.
