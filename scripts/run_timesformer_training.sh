#!/bin/bash
# =============================================================================
# Run TimeSformer-base training on a GPU instance (V100-32GB or better)
#
# Usage:
#   bash scripts/run_timesformer_training.sh                   # defaults: bs=8, epochs=15
#   bash scripts/run_timesformer_training.sh --batch-size 16   # if VRAM allows
#   bash scripts/run_timesformer_training.sh --num-frames 16   # higher accuracy, ~2x VRAM
#   bash scripts/run_timesformer_training.sh --resume models/timesformer/checkpoints/last.pt
#
# Pretrained weights: facebook/timesformer-base-finetuned-k400 (~86M params)
# Downloaded automatically from HuggingFace on first run (~330 MB).
#
# Outputs:
#   results/TimeSformer_results.json           ← top-1, top-5, F1, per-class, params, time, VRAM
#   results/TimeSformer_confusion_matrix.npy   ← 174x174 confusion matrix
#   models/timesformer/checkpoints/best.pt     ← best val_acc weights
#   models/timesformer/checkpoints/last.pt     ← latest (resume from here)
#   models/timesformer/checkpoints/metrics.csv ← per-epoch log
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Resolve Python: prefer the active env if it has torch+transformers,
# then the cap5610 conda env, then the Vast.ai venv.
_pick_python() {
    if python -c "import torch; import transformers" &>/dev/null 2>&1; then
        echo python; return
    fi
    local conda_base
    conda_base=$(conda info --base 2>/dev/null)
    if [ -n "$conda_base" ] && [ -f "${conda_base}/envs/cap5610/bin/python" ]; then
        echo "${conda_base}/envs/cap5610/bin/python"; return
    fi
    if [ -f /venv/main/bin/python ]; then
        echo /venv/main/bin/python; return
    fi
    echo "ERROR: torch/transformers not found. Activate your environment (e.g. conda activate cap5610)." >&2
    exit 1
}
PYTHON=$(_pick_python)

"$PYTHON" models/timesformer/train.py \
    --epochs 15 \
    --batch-size 8 \
    --lr 1e-4 \
    --num-frames 8 \
    --num-workers 8 \
    --mixed-precision bf16 \
    "$@"
