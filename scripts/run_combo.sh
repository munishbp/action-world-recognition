#!/usr/bin/env bash
# Run two stages sequentially:
#   1. VideoMamba SSv2-pretrained eval         -> Tier-1 SOTA reference baseline
#   2. CNN+ConvLSTM 5-epoch training run        -> own learning curve for analysis
#
# Both stages write standard results via shared.evaluate_model + save_results:
#   results/VideoMambaSSv2Pretrained_results.json (+ confusion matrix .npy)
#   results/CNNConvLSTM_results.json              (+ confusion matrix .npy)
#
# Per-epoch metrics for the CNN+ConvLSTM learning curve:
#   models/cnn_convlstm/checkpoints/metrics.csv
#
# Usage (from repo root):
#   bash scripts/run_combo.sh
#
# Env-var overrides:
#   SKIP_EVAL=1     bash scripts/run_combo.sh   # skip stage 1
#   SKIP_TRAIN=1    bash scripts/run_combo.sh   # skip stage 2
#   EPOCHS=10       bash scripts/run_combo.sh   # adjust training epochs
#   TRAIN_BATCH=64  bash scripts/run_combo.sh   # raise CNN+ConvLSTM batch size
#   EVAL_BATCH=48   bash scripts/run_combo.sh   # raise VideoMamba eval batch size
#   NUM_WORKERS=12  bash scripts/run_combo.sh   # dataloader workers

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
EPOCHS="${EPOCHS:-5}"
EVAL_BATCH="${EVAL_BATCH:-32}"
TRAIN_BATCH="${TRAIN_BATCH:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"

LOG_DIR="$REPO_ROOT/results/run_logs"
mkdir -p "$LOG_DIR"
EVAL_LOG="$LOG_DIR/videomamba_eval_$(date +%Y%m%d_%H%M%S).log"
TRAIN_LOG="$LOG_DIR/cnn_convlstm_train_$(date +%Y%m%d_%H%M%S).log"

START_TS=$(date +%s)

banner() { echo; echo "=========================================="; echo "$1"; echo "=========================================="; }

# --- Stage 1: VideoMamba SSv2-pretrained eval ---
if [[ "$SKIP_EVAL" != "1" ]]; then
    banner "Stage 1/2: VideoMamba SSv2-pretrained eval"
    CKPT="models/videomamba/checkpoints/videomamba_s16_ssv2_f16_res224.pth"
    if [[ ! -f "$CKPT" ]]; then
        echo "Checkpoint missing -- downloading..."
        bash models/videomamba/download_ssv2_weights.sh small 16
    fi
    echo "Logging to: $EVAL_LOG"
    python models/videomamba/eval.py \
        --model small \
        --num-frames 16 \
        --ckpt "$CKPT" \
        --batch-size "$EVAL_BATCH" \
        --num-workers "$NUM_WORKERS" \
        2>&1 | tee "$EVAL_LOG"
    EVAL_DONE=$(date +%s)
    echo "Stage 1 elapsed: $(( (EVAL_DONE - START_TS) / 60 )) min"
else
    echo "SKIP_EVAL=1 -- skipping VideoMamba eval"
fi

# --- Stage 2: CNN+ConvLSTM training ---
if [[ "$SKIP_TRAIN" != "1" ]]; then
    banner "Stage 2/2: CNN+ConvLSTM training (${EPOCHS} epochs)"
    echo "Logging to: $TRAIN_LOG"
    python models/cnn_convlstm/train.py \
        --epochs "$EPOCHS" \
        --batch-size "$TRAIN_BATCH" \
        --num-frames 8 \
        --num-workers "$NUM_WORKERS" \
        --lr 1e-3 \
        2>&1 | tee "$TRAIN_LOG"
    TRAIN_DONE=$(date +%s)
    echo "Stage 2 elapsed: $(( (TRAIN_DONE - START_TS) / 60 )) min from start"
else
    echo "SKIP_TRAIN=1 -- skipping CNN+ConvLSTM training"
fi

banner "All done"
TOTAL=$(( $(date +%s) - START_TS ))
printf "Total elapsed: %d min %d sec\n" $((TOTAL / 60)) $((TOTAL % 60))
echo
echo "Results:"
ls -la results/VideoMambaSSv2Pretrained_results.json results/VideoMambaSSv2Pretrained_confusion_matrix.npy 2>/dev/null || true
ls -la results/CNNConvLSTM_results.json results/CNNConvLSTM_confusion_matrix.npy 2>/dev/null || true
echo
echo "Per-epoch CNN+ConvLSTM metrics (for learning-curve analysis):"
ls -la models/cnn_convlstm/checkpoints/metrics.csv 2>/dev/null || true
echo
echo "Logs:"
ls -la "$LOG_DIR"/*.log 2>/dev/null || true
