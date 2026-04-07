#!/bin/bash
# =============================================================================
# Run VideoMamba training on Vast.ai A100
#
# Usage:
#   bash scripts/run_training.sh              # defaults: batch=32, epochs=30
#   bash scripts/run_training.sh --batch-size 64
#   bash scripts/run_training.sh --resume models/videomamba/checkpoints/last.pt
#
# Batch size strategy (A100 40GB, fp16):
#   Try 64 → OOM? try 32 → OOM? try 16
#   Larger batch = fewer steps = less GPU time = saves money
#
# Results are saved to:
#   results/VideoMamba_results.json       ← metrics (top1, top5, F1, etc.)
#   results/VideoMamba_confusion_matrix.npy
#   models/videomamba/checkpoints/best.pt ← best model weights
#   models/videomamba/checkpoints/last.pt ← latest checkpoint (for resuming)
#   models/videomamba/checkpoints/metrics.csv ← per-epoch log
#
# After training, download results/ and checkpoints/best.pt from Vast.ai.
# Validation runs automatically after every epoch — val accuracy is your
# final reported metric (SSv2 test set labels are not public).
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python models/videomamba/train.py \
    --model tiny \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --num-frames 16 \
    --num-workers 8 \
    "$@"
