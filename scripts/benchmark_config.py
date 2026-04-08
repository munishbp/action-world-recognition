"""
Benchmark different batch_size / num_workers combos to find the fastest config.

Usage:
    python scripts/benchmark_config.py

Runs 50 batches per config, measures samples/sec, prints a ranked table.
"""

import os
import sys
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

import torch
from shared import get_dataloader

DATA_ROOT = os.path.join(_PROJECT_ROOT, "data", "something-something-v2")
NUM_FRAMES = 16
NUM_BATCHES = 50  # batches per config

BATCH_SIZES   = [32, 64, 128]
WORKER_COUNTS = [8, 16, 32]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []

print(f"Benchmarking {len(BATCH_SIZES) * len(WORKER_COUNTS)} configs x {NUM_BATCHES} batches each...\n")

for bs in BATCH_SIZES:
    for nw in WORKER_COUNTS:
        try:
            loader = get_dataloader(
                split="train",
                batch_size=bs,
                num_frames=NUM_FRAMES,
                num_workers=nw,
                pin_memory=(device.type == "cuda"),
                root=DATA_ROOT,
            )

            # Warm up
            it = iter(loader)
            for _ in range(3):
                batch = next(it)

            # Timed run
            start = time.perf_counter()
            samples = 0
            for i, batch in enumerate(it):
                if i >= NUM_BATCHES:
                    break
                if batch is None:
                    continue
                frames, labels = batch
                frames = frames.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                samples += labels.size(0)
            elapsed = time.perf_counter() - start

            sps = samples / elapsed
            results.append((sps, bs, nw))
            print(f"  batch={bs:3d}  workers={nw:2d}  →  {sps:7.1f} samples/sec")

        except Exception as e:
            print(f"  batch={bs:3d}  workers={nw:2d}  →  FAILED: {e}")

results.sort(reverse=True)
print("\n=== Ranked Results ===")
for rank, (sps, bs, nw) in enumerate(results, 1):
    epoch_hours = (168913 / sps) / 3600
    print(f"  #{rank}  batch={bs:3d}  workers={nw:2d}  {sps:7.1f} samples/sec  (~{epoch_hours:.1f}h/epoch)")

print(f"\nBest config: --batch-size {results[0][1]} --num-workers {results[0][2]}")
