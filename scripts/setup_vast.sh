#!/bin/bash
# =============================================================================
# Vast.ai A100 Setup Script — VideoMamba on Something-Something V2
#
# Tested image: vastai/pytorch:2.11.0-cu128-cuda-12.9-mini-py311-2026-03-26
#
# Run from the repo root:
#   bash scripts/setup_vast.sh
#
# What this does:
#   1. Installs all Python dependencies
#   2. Installs mamba-ssm via pre-built wheel (mini image has no nvcc)
#   3. Fixes VideoMamba __init__.py
#   4. Verifies imports (smoke test)
#   5. Downloads the SSv2 dataset (~20GB)
#   6. Runs a 2-batch end-to-end training smoke test
# =============================================================================

set -e  # exit immediately on any error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
echo "=== Repo root: $REPO_ROOT ==="

# ── 1. Dependencies ───────────────────────────────────────────────────────────
echo ""
echo "=== [1/6] Installing dependencies ==="

pip install --quiet numpy
# Downgrade torch to 2.7 — required to match the mamba-ssm pre-built wheel
# (mini image has no nvcc so we can't build from source; latest wheel is torch2.7)
pip install --quiet torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install --quiet -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install --quiet timm einops huggingface_hub triton gdown

echo "Dependencies installed."

# ── 2. Install mamba-ssm via pre-built wheel ──────────────────────────────────
echo ""
echo "=== [2/6] Installing mamba-ssm (pre-built wheel) ==="
# Mini images have no nvcc so building from source is not possible.
# Pre-built wheel: torch2.7, Python 3.11, cxx11abiTRUE, Linux x86_64
WHEEL_URL="https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu11torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
pip install --quiet "$WHEEL_URL"
# The wheel was built against CUDA 11 so it needs libcudart.so.11.0 at runtime
pip install --quiet nvidia-cuda-runtime-cu11

# Add the CUDA 11 runtime lib to LD_LIBRARY_PATH so the linker can find it
CUDA_RT_LIB=$(python -c "import os, nvidia.cuda_runtime; print(os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), 'lib'))")
export LD_LIBRARY_PATH="$CUDA_RT_LIB:$LD_LIBRARY_PATH"
# Persist for future shell sessions
echo "export LD_LIBRARY_PATH=\"$CUDA_RT_LIB:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
echo "mamba-ssm installed."

# ── 3. Fix VideoMamba __init__.py files ──────────────────────────────────────
echo ""
echo "=== [3/6] Fixing VideoMamba __init__.py ==="

touch "$REPO_ROOT/models/__init__.py"
touch "$REPO_ROOT/models/videomamba/__init__.py"
echo "# Minimal init for class project" > "$REPO_ROOT/models/videomamba/models/__init__.py"
echo "Fixed __init__.py files."

# ── 4. Verify imports ─────────────────────────────────────────────────────────
echo ""
echo "=== [4/6] Verifying imports ==="

python -c "
import sys
sys.path.insert(0, '.')
results = {}
for mod in ['videomamba', 'modeling_finetune', 'deit']:
    try:
        exec(f'from models.videomamba.models import {mod}')
        results[mod] = 'OK'
    except Exception as e:
        results[mod] = f'FAIL: {e}'

for mod, status in results.items():
    print(f'  {mod}: {status}')

failures = [m for m, s in results.items() if 'FAIL' in s]
if failures:
    print(f'IMPORT FAILURES: {failures}')
    sys.exit(1)
else:
    print('All imports OK.')
"

# ── 5. Download dataset ───────────────────────────────────────────────────────
echo ""
echo "=== [5/6] Downloading SSv2 dataset (~20GB) ==="
echo "This will take several minutes depending on bandwidth..."

python scripts/download_dataset.py

# ── 6. Smoke test: 2-batch end-to-end training run ───────────────────────────
echo ""
echo "=== [6/6] Running end-to-end smoke test (2 batches) ==="

python models/videomamba/train.py \
    --smoke-test \
    --batch-size 4 \
    --num-frames 16 \
    --num-workers 2

echo ""
echo "============================================="
echo " Setup complete! Everything is working."
echo ""
echo " To start full training:"
echo "   bash scripts/run_training.sh"
echo "============================================="
