# Persistent PyTorch speech worker

One process keeps one GPU's BF16 Whisper encoder, BF16 speech projector, BF16 Qwen text model and session caches resident. The trained FP32 projector weights are cast to BF16 at load time. PyTorch retains higher-precision accumulation and Qwen's FP32 recurrent state where needed. Rust selects every batch and accepts every output token. The worker performs complete-utterance encoding/prefill or one decode step; it never starts an independent generation loop. The public application's audio and text remain in the Rust session record. Device count is determined by the number of configured worker endpoints.

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

CPU tests run the actual Transformers 5.13 Qwen hybrid model with small random weights, not a mock cache. They compare attention, convolution and recurrent state and logits for ragged decode batches, chunked multimodal-style continuations, changing membership and interruption reconciliation. Preparation tests verify committed cache tensors stay bit-identical and activation matches ordinary prefill over multiple turns. Test-only model and CUDA-memory fixtures exercise the production engine's dispatch and validation. GPU integration tests are marked and skipped unless `VOICE_WORKER_CONFIG` and a CUDA device are available.

Validated local lockfile: Python 3.12.13, PyTorch 2.14.1 CPU, Transformers 5.13.0, Pydantic 2.13.5. Python 3.10 or newer is supported by the project metadata; numerical checks were run on Python 3.12. PyTorch's CPU execution is used solely for these tests; the production worker requires CUDA.

The [shared RTX 3090 deployment](../docs/DEPLOYMENT_3090.md) additionally validates an intermediate 10 Hz checkpoint with Python 3.12.14, PyTorch 2.6.0+cu124 and the node's existing fast kernels. Its separate environment reads existing CUDA packages without upgrading the training environment. The supported PyTorch minimum is 2.6; the local lockfile remains unchanged apart from that requirement.

## Linux GPU deployment

Use a CUDA-capable Linux node with sufficient driver support for the locked PyTorch wheel. The worker uses the node's existing driver; it does not install or change it. Use one worker configuration and process per selected device. The [current hardware report](../docs/GPU_IMMEDIATE_COMMIT_3090.md) measures one RTX 3090; physical multi-GPU execution remains unbenchmarked.

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

   Keep worker processes under the deployment's process supervisor. Model snapshots download at their manifest-pinned revisions; a cached Hugging Face snapshot avoids repeated downloads. Startup verifies the checkpoint digest, loads the text-only Qwen class and only Whisper's encoder onto the GPU, sets evaluation/inference mode, verifies prompt token IDs and EOS, and warms complete audio prefill plus cached decode before the listening socket opens. Readiness is a framed capability message after a gateway connects. `warmup` configures representative audio length, prefill lengths and maximum warmed prefill/decode batches. Defaults use six seconds of audio, 64/256-position prompts and batches up to four, clamped to worker limits. Every encoder/prefill batch size from one through that limit is warmed. Cached prefill covers the two-token completed-turn delimiter and three-token interrupted-turn reconciliation. Decode warms one, powers of two and its configured upper size using a single populated source cache; transient warmup caches are freed before listening.

5. Run hardware cache tests before accepting capacity claims:

   ```bash
   VOICE_WORKER_CONFIG=/absolute/path/to/worker-0.json uv run pytest -q -m integration
   ```

   Repeat for every configured GPU, including nonzero device indices. GPU CUDA events explicitly use the model's device stream. Compare the full speech route against saved reference audio/features and next-token logits from training, then run the public WebSocket workload runner. The included integration tests compare BF16 cached continuation/full replay and ragged batching/serial logits using complete synthetic audio; they do not establish speech quality or replace reference-checkpoint parity testing.

## Cache and batching invariants

Each session owns a Transformers `DynamicCache` configured for the pinned Qwen model. All full-attention K/V and linear-attention convolution/recurrent tensors are retained. No attention-only cropping, rollback or generic batch-selection method is used.

`prepare` runs complete-utterance encoding and prefill into one provisional session branch. It shares the original cache only until the model joins cloned tensors for its forward. The committed cache, token acceptance and text preview remain unchanged. The resulting first token stays provisional until matching `activate` promotes the branch and returns that already computed token without another forward. Turn and generation must match, and the source cache must still be current. `discard_prepared` frees the optional branch; resumed audio can then prepare a new complete snapshot. Normal prefill, decode, close and connection reset invalidate any held preparation. Prepare and ordinary prefill remain distinct homogeneous worker batches.

Each prepared branch reserves one additional maximum-context cache allocation. Admission and transient workspace checks include its actual materialized bytes and unmaterialized reservation. Speculation can return `capacity_exceeded` while ordinary prefill remains possible. A preparation failure releases only provisional state and retains committed conversation history; failed ordinary forwards still invalidate affected committed sessions. No provisional output is part of the accepted session history.

Decode batches can contain sessions with different context lengths. Joining left-pads **attention tensors only**, builds a matching attention mask, and supplies each row's true absolute positions. Fixed-size convolution/recurrent state is concatenated along the batch dimension. Splitting removes only temporary attention padding and clones every session's state so changing batches cannot alias another session's tensors.

Prefill first encodes audio as one Whisper batch, projects each real cropped sequence before padding, and groups language-model forwards by new prompt length and fresh/populated cache state. Each group executes a real batched forward. This avoids corrupting recurrent state by padding new tokens. Different speech lengths can therefore produce several forwards within one Rust-selected request; timings include all groups. There is no permanent batch membership, and no assumption of linear cost versus batch fill.

Rust sends `AcceptedToken {turn_id, index, token_id}` with decode and optionally with the next prefill. A forward consumes only that accepted token and proposes the next token. Proposal bytes and detokenizer state are committed only after matching Rust acceptance. Equal token IDs at successive positions remain distinct through the index. A new user turn reconciles a pending final accepted token exactly once, closes the assistant message once with EOS, inserts the newline, then adds the trained user prefix, speech embeddings and assistant suffix. An in-flight proposal rejected after interruption is never consumed. Generation epochs must increase; public turn identifiers need not be numerically ordered.

Byte-level detokenization buffers incomplete UTF-8 characters. Invalid byte sequences produce replacement text because arbitrary generated byte tokens need not form valid UTF-8. Each proposal previews this decoding; only acceptance commits the incomplete tail. EOS flushes any previously accepted incomplete bytes with replacement text. Cancelling or hitting a token limit can end before a complete UTF-8 character exists; exact accepted IDs remain recorded, while the incomplete character is not invented or sent. The trained suffix already opens the assistant role with an empty thinking block; there is no additional audio or BOS token.

EOS itself is an accepted, recorded model token, including when its text delta is empty. Generated model-token counts include EOS; the fixture's configured ordinary response tokens are followed by one EOS token. A later prefill consumes that pending accepted EOS without inserting a second assistant-end delimiter.

Session admission reserves the estimated maximum configured context's attention KV plus FP32 recurrent and BF16 convolution state against `cache_budget_bytes`. Actual free device memory must also cover every admitted session's unmaterialized reservation, the new session and `workspace_reserve_bytes`; already materialized cache tensors are excluded from this additional charge. Each forward separately reserves conservative join/split workspace using the longest padded context and appended positions for every batch row. These checks preserve remaining session reservations while transient copies coexist. `max_sessions` also gates allocation. The reserve is conservative and fixed per admitted session; it is not a paged allocator or a measured throughput guarantee. Free device memory excludes PyTorch's unused allocator blocks, so these checks can conservatively reject a batch that might fit through allocator reuse. Attention state grows only to `max_context_tokens`; context exhaustion is explicit and never silently truncates history. Closing a session or losing the gateway connection releases owned caches. A failed model forward invalidates affected sessions' caches and reports `backend_failed`.

`allocator_memory_fraction` defaults to 1.0 and caps this process's PyTorch CUDA allocator. Admission and workspace checks also respect the remaining quota after allocator-reserved memory, even if other device memory is free. This does not cap allocations made directly by external CUDA libraries or partition the GPU against another process. The shared-node smoke deployment uses 0.25, one session and a resource guard; its guard tool targets the inspected Linux cgroup-v1 host and stops only its owned child.

## Measurement and remaining assumptions

Forward timings use CUDA events on the device stream, including cache join/split. Returning scalar proposed token IDs already synchronizes the completed batch, so timing does not add another per-token device synchronization. These are completed device-stream elapsed times, including host launch gaps, rather than SM busy time. `encode_ms` starts after CPU Whisper feature extraction and includes host-to-device transfer, encoder and projector execution. CPU PCM conversion and mel preprocessing are included in the RPC's total `elapsed_ms`; the wire protocol does not expose a separate CPU preprocessing phase. Responses report actual CUDA allocated and allocator-reserved bytes.

The implementation currently copies hybrid tensors when joining and splitting dynamic batches. This provides simple ownership and isolation, but its allocation/bandwidth overhead must be profiled on the rented node. Left-padding very unequal contexts also adds attention work. Neither cost has a hardware performance claim. Equal-new-length prefill grouping may reduce fill for irregular audio durations; report both Rust-selected and actual model-group batch sizes when profiling.

Transformers supports a pure PyTorch fallback for Qwen's linear attention when optional fast kernels are absent. The lockfile intentionally uses that supported path initially; kernel optimization, CUDA graphs, paged caches and packed variable-length prefills require separate parity and speed validation. Bounded representative warmup does not eliminate every later batch/context shape's first-use overhead.

The retained interleaved speech/text history is the agreed serving target and a training-owned model-quality assumption. The runtime supports it with persistent hybrid state. The [immediate-commit benchmark](../docs/GPU_IMMEDIATE_COMMIT_3090.md) records real audio TTFT, token gaps, rolling generation rates, stage costs and sampled RAM/VRAM. Longer varied conversations, external networks, physical multi-GPU execution and capacity tuning remain separate validation work. The old audio-echo benchmark is unrelated to this model's capacity.

The strict version-one metadata and PCM wire contract is documented in the repository's `docs/WORKER_PROTOCOL.md`; `protocol.py` mirrors the Rust types. Only one gateway connection owns a worker at a time. Frames, bodies, active caches and transport wait durations have bounds; model forwards run on one dedicated thread so Python's network event loop remains responsive.

A healthy gateway connection and its session caches may remain idle indefinitely. The transport timeout starts when the first byte of a new frame arrives, then bounds the complete remaining header, metadata and audio body with one shared budget. It also bounds response backpressure. Idle waiting for the next frame's first byte has no deadline; disconnect or worker shutdown still cancels that wait and releases every owned cache.
