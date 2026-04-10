#!/bin/bash
# =============================================================================
# Vast.ai Setup Script — SlowFast-R50 on Something-Something V2
#
# Tested image: vastai/pytorch:2.11.0-cu128-cuda-12.9-mini-py311-2026-03-26
#
# Run from the repo root:
#   bash scripts/setup_vast_slowfast.sh
#
# What this does:
#   1. Installs all Python dependencies
#   2. Verifies imports and model load (smoke test)
#   3. Downloads the SSv2 dataset (~20GB)
# =============================================================================

set -e  # exit immediately on any error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
echo "=== Repo root: $REPO_ROOT ==="

# ── 1. Dependencies ───────────────────────────────────────────────────────────
echo ""
echo "=== [1/3] Installing dependencies ==="

# Use the PyTorch already installed in the image — do not downgrade or reinstall it.
# (Downgrading to cu126 causes "no kernel image" errors on cu128/cu129 images.)
pip install --quiet -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
# pytorchvideo (used by torch.hub for SlowFast) requires fvcore and iopath
pip install --quiet fvcore iopath gdown

echo "Dependencies installed."

# ── 2. Verify imports and model load ─────────────────────────────────────────
echo ""
echo "=== [2/3] Verifying imports and SlowFast model load ==="

python -c "
import sys
sys.path.insert(0, '.')

print('  Checking shared pipeline...')
from shared.dataset import SomethingSomethingV2Dataset
from shared import evaluate_model, save_results
print('  shared: OK')

print('  Loading SlowFast-R50 from torch.hub (downloads ~100MB on first run)...')
import torch
import torch.nn as nn
hub = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
head = hub.blocks[-1]
head.proj = nn.Linear(head.proj.in_features, 174)
print(f'  SlowFast-R50: OK (head -> 174 classes)')
"

echo "Imports OK."

# ── 3. Download dataset ───────────────────────────────────────────────────────
echo ""
echo "=== [3/3] Downloading SSv2 dataset (~20GB) ==="
echo "This will take several minutes depending on bandwidth..."

python scripts/download_dataset.py

echo ""
echo "============================================="
echo " Setup complete! Everything is working."
echo ""
echo " To start training:"
echo "   python models/SlowFast/slowfast.py"
echo ""
echo " Common options:"
echo "   --epochs 20 --batch-size 8 --lr 0.01"
echo "   --resume models/SlowFast/checkpoints/last.pt"
echo "============================================="
