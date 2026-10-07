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

The original shared-training configuration capped the PyTorch allocator at 25% of device memory, with one session and a 128 MiB cache budget. After training and evaluation finished, the [idle-node benchmark](../../docs/GPU_BENCHMARK_3090.md) used a separate completed checkpoint and bounded larger limits. Production `WorkerConfig` defaults remain separate from either experiment.

The guard handles a child exiting during `/proc` sampling without concealing a missing resource statistic for a live child. Run its six regression cases with `uv run pytest tools/test_guard_worker.py` from `backend/`; include that path alongside `tests` for the full operator/backend suite.
