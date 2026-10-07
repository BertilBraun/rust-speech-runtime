# Shared-node deployment helpers

These operator tools were used for the [2026-10-07 RTX 3090 smoke deployment](../../docs/DEPLOYMENT_3090.md). They do not alter training or manage the rented instance itself.

`snapshot_checkpoint.py` copies a completed atomic projector save into a new directory, preserves its training configuration/state and creates a canonical `WorkerConfig` with the copied checksum. It refuses an existing destination or a save that changes during capture. It copies neither optimizer state nor the language-model/encoder snapshots. Before using it, verify the source configuration matches this adapter: mean pooling by five, MLP 768 → 1,024 → 2,048, Whisper Small, Qwen3.5-2B, chat without a system prompt and greedy decoding. The helper records training metadata; it does not discover a different model architecture.

```bash
PYTHONPATH=backend/src .venv/bin/python backend/tools/snapshot_checkpoint.py \
  --source-run /absolute/path/to/training-run \
  --destination /absolute/path/to/new-serving-checkpoint
```

`guard_worker.py` starts one owned child at low CPU priority, writes resource observations every two seconds and terminates that child if a configured threshold is crossed. It monitors GPU zero and the inspected node's Linux cgroup-v1 memory files, counting inactive file cache as reclaimable. It does not signal training processes or change global kernel settings. This is a best-effort monitor, not resource partitioning: another process can allocate between observations, direct CUDA-library allocations are outside PyTorch's allocator cap, and child RSS does not include every descendant's memory. Keep workloads small on a shared training GPU.

```bash
.venv/bin/python backend/tools/guard_worker.py \
  --report worker-resources.jsonl \
  --minimum-free-vram-mib 4096 \
  --minimum-available-ram-mib 4096 \
  --maximum-child-rss-mib 6144 \
  -- .venv/bin/python -m voice_worker.cli --config /absolute/path/to/worker.json
```

The serving configuration additionally caps the PyTorch allocator at 25% of device memory, with one session, batch size one, context 2,048 and 128 MiB cache budget. Production `WorkerConfig` defaults remain separate from these deliberately small shared-node limits.
