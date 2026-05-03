#!/usr/bin/env bash
# Self-contained setup + run for Box 1: VideoMamba SSv2-pretrained eval.
#
# What this does:
#   1. Installs pinned Python dependencies
#   2. Sets up SSv2 annotations and downloads the ~220K-clip dataset
#      (skips automatically if already present)
#   3. Downloads the SSv2-finetuned VideoMamba-Small @ 16 frames checkpoint
#   4. Runs eval-only on the val split
#
# Outputs:
#   results/VideoMambaSSv2Pretrained_results.json
#   results/VideoMambaSSv2Pretrained_confusion_matrix.npy
#
# Usage (from repo root, after `git pull`):
#   bash scripts/box_videomamba_eval.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EVAL_BATCH="${EVAL_BATCH:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"

banner() { echo; echo "=========================================="; echo "$1"; echo "=========================================="; }

banner "Step 1/4: Installing Python dependencies"
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install gdown huggingface_hub datasets

banner "Step 2/4: SSv2 dataset setup"
python scripts/download_dataset.py

banner "Step 3/4: Downloading VideoMamba SSv2-pretrained checkpoint"
bash models/videomamba/download_ssv2_weights.sh small 16

banner "Step 4/4: Running VideoMamba eval on SSv2 val"
LOG_DIR="$REPO_ROOT/results/run_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/videomamba_eval_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG"

python models/videomamba/eval.py \
    --model small \
    --num-frames 16 \
    --ckpt models/videomamba/checkpoints/videomamba_s16_ssv2_f16_res224.pth \
    --batch-size "$EVAL_BATCH" \
    --num-workers "$NUM_WORKERS" \
    2>&1 | tee "$LOG"

banner "Done"
echo "Results:"
ls -la results/VideoMambaSSv2Pretrained_results.json results/VideoMambaSSv2Pretrained_confusion_matrix.npy 2>/dev/null || true
echo "Log: $LOG"
