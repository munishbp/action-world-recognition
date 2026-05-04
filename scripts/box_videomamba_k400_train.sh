#!/usr/bin/env bash
# Self-contained launcher for VideoMamba-Small bidirectional fine-tuning
# from the K400-pretrained checkpoint on SSv2.
#
# Assumes the SSv2 dataset is already in place at
#   data/something-something-v2/  (rsync from another box, or run download_dataset.py first)
#
# Usage (from repo root, after `git pull`):
#   bash scripts/box_videomamba_k400_train.sh
#
# Env-var overrides:
#   EPOCHS=15        TRAIN_BATCH=16   NUM_FRAMES=16
#   NUM_WORKERS=16   LR=1e-4          INIT_KIND=k400   (or in1k)
#   MODEL_NAME=VideoMambaK400Finetuned
#   SMOKE_TEST=1     # run a 2-batch smoke test before committing to full training

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

EPOCHS="${EPOCHS:-15}"
TRAIN_BATCH="${TRAIN_BATCH:-16}"
NUM_FRAMES="${NUM_FRAMES:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LR="${LR:-1e-4}"
INIT_KIND="${INIT_KIND:-k400}"
MODEL_NAME="${MODEL_NAME:-VideoMambaK400Finetuned}"
SMOKE_TEST="${SMOKE_TEST:-0}"

case "$INIT_KIND" in
    k400) INIT_FILENAME="videomamba_s16_k400_f16_res224.pth" ;;
    in1k) INIT_FILENAME="videomamba_s16_in1k_res224.pth" ;;
    *) echo "Unknown INIT_KIND: $INIT_KIND (expected k400|in1k)"; exit 1 ;;
esac
INIT_PATH="$REPO_ROOT/models/videomamba/checkpoints/$INIT_FILENAME"

banner() { echo; echo "=========================================="; echo "$1"; echo "=========================================="; }

banner "Step 1/4: Verifying SSv2 dataset"
DATA_ROOT="$REPO_ROOT/data/something-something-v2"
if [[ ! -d "$DATA_ROOT/annotations" ]]; then
    echo "Annotations missing at $DATA_ROOT/annotations"
    echo "Either rsync the dataset from another box or run scripts/download_dataset.py first."
    exit 1
fi
WEBM_COUNT=$(find "$DATA_ROOT" -maxdepth 1 -name "*.webm" 2>/dev/null | wc -l)
echo "Found $WEBM_COUNT .webm files at $DATA_ROOT"
if [[ "$WEBM_COUNT" -lt 100000 ]]; then
    echo "Suspiciously few webm files — expected ~220,000. Aborting."
    exit 1
fi

banner "Step 2/4: Verifying Python deps"
python -c "import torch; import einops; import timm; from mamba_ssm.modules.mamba_simple import Mamba; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

banner "Step 3/4: Downloading pretrained weights ($INIT_KIND)"
if [[ ! -f "$INIT_PATH" ]]; then
    bash scripts/download_videomamba_pretrained.sh "$INIT_KIND"
else
    echo "Already present: $INIT_PATH"
fi

banner "Step 4/4: Training VideoMamba-Small bidir from $INIT_KIND init"
LOG_DIR="$REPO_ROOT/results/run_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/videomamba_${INIT_KIND}_train_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG"

SMOKE_FLAG=""
if [[ "$SMOKE_TEST" == "1" ]]; then
    SMOKE_FLAG="--smoke-test"
    echo "[SMOKE TEST MODE] Will run 2 train + 2 val batches only."
fi

python models/videomamba/train.py \
    --bidir \
    --model small \
    --num-frames "$NUM_FRAMES" \
    --epochs "$EPOCHS" \
    --batch-size "$TRAIN_BATCH" \
    --num-workers "$NUM_WORKERS" \
    --lr "$LR" \
    --init-from "$INIT_PATH" \
    --model-name "$MODEL_NAME" \
    $SMOKE_FLAG \
    2>&1 | tee "$LOG"

banner "Done"
ls -la "results/${MODEL_NAME}_results.json" "results/${MODEL_NAME}_confusion_matrix.npy" 2>/dev/null || true
echo "Per-epoch metrics: models/videomamba/checkpoints/metrics.csv"
echo "Log: $LOG"
