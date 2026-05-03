#!/usr/bin/env bash
# Self-contained setup + run for Box 2: CNN+ConvLSTM 5-epoch training.
#
# What this does:
#   1. Installs pinned Python dependencies
#   2. Sets up SSv2 annotations and downloads the ~220K-clip dataset
#      (skips automatically if already present)
#   3. Trains CNN+ConvLSTM for 5 epochs on SSv2 train, evaluating on val each
#      epoch, and runs a final eval on the best checkpoint.
#
# Outputs:
#   results/CNNConvLSTM_results.json
#   results/CNNConvLSTM_confusion_matrix.npy
#   models/cnn_convlstm/checkpoints/metrics.csv  (per-epoch learning curve)
#   models/cnn_convlstm/checkpoints/best.pt
#   models/cnn_convlstm/checkpoints/last.pt
#
# Usage (from repo root, after `git pull`):
#   bash scripts/box_cnn_convlstm_train.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EPOCHS="${EPOCHS:-5}"
TRAIN_BATCH="${TRAIN_BATCH:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-1e-3}"

banner() { echo; echo "=========================================="; echo "$1"; echo "=========================================="; }

banner "Step 1/3: Installing Python dependencies"
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install gdown huggingface_hub datasets

banner "Step 2/3: SSv2 dataset setup"
python scripts/download_dataset.py

banner "Step 3/3: Training CNN+ConvLSTM for ${EPOCHS} epochs"
LOG_DIR="$REPO_ROOT/results/run_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/cnn_convlstm_train_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG"

python models/cnn_convlstm/train.py \
    --epochs "$EPOCHS" \
    --batch-size "$TRAIN_BATCH" \
    --num-frames 8 \
    --num-workers "$NUM_WORKERS" \
    --lr "$LR" \
    2>&1 | tee "$LOG"

banner "Done"
echo "Results:"
ls -la results/CNNConvLSTM_results.json results/CNNConvLSTM_confusion_matrix.npy 2>/dev/null || true
echo
echo "Per-epoch metrics (learning curve):"
ls -la models/cnn_convlstm/checkpoints/metrics.csv 2>/dev/null || true
echo "Log: $LOG"
