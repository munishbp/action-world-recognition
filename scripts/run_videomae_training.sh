#!/bin/bash
# =============================================================================
# Run VideoMAE-base training on Vast.ai RTX Pro 6000 WS (96 GB, Blackwell sm_120)
#
# Usage:
#   bash scripts/run_videomae_training.sh                  # defaults: bs=64, epochs=30
#   bash scripts/run_videomae_training.sh --batch-size 32  # if OOM at 64
#   bash scripts/run_videomae_training.sh --resume models/videomae/checkpoints/last.pt
#
# Blackwell note:
#   Use a PyTorch image built against CUDA 12.8+ (e.g. vastai/pytorch:2.7.0-cuda-12.8.0).
#   Stable PyTorch < 2.6 will throw "no kernel image" on sm_120 (see SlowFast notes
#   in RESULTS.md). Verify before launching:
#     python -c "import torch; x=torch.randn(8,device='cuda'); print((x@x).item())"
#
# Outputs:
#   results/VideoMAE_results.json           ← top-1, top-5, F1, per-class, params, time, VRAM
#   results/VideoMAE_confusion_matrix.npy   ← 174x174 confusion matrix
#   models/videomae/checkpoints/best.pt     ← best val_acc weights
#   models/videomae/checkpoints/last.pt     ← latest (resume from here)
#   models/videomae/checkpoints/metrics.csv ← per-epoch log
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v python &>/dev/null; then
    source /venv/main/bin/activate
fi

python models/videomae/train.py \
    --epochs 30 \
    --batch-size 64 \
    --lr 5e-4 \
    --num-frames 16 \
    --num-workers 8 \
    --mixed-precision bf16 \
    "$@"
