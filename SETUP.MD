# VideoMamba + Project Setup Guide

This guide supports two environments:
- **Vast.ai A100 PCIE** (40GB VRAM) — primary training environment
- **Local WSL** (Windows + WSL2) — for development/testing only

---

# Vast.ai Instance Setup

When renting on Vast.ai:
- **GPU**: A100 PCIE (40GB)
- **Image**: `vastai/pytorch:2.5.1-cuda-12.4.1` (search this in the Vast.ai template picker)
- **Disk**: at least 80GB (dataset + model weights)
- **On-Start Script**: leave blank, follow this guide manually

---

# 1. Install Conda (if not already installed)

Skip this step on cloud GPU instances — conda is usually pre-installed.

Run this in your terminal:

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow prompts:

* Press ENTER
* Type `yes`
* Accept default install location

Then restart shell OR run:

```bash
source ~/.bashrc
```

Verify:

```bash
conda --version
```

---

# 2. Create Project Environment

On first run, conda may require ToS acceptance:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Create and activate the environment:

```bash
conda create -n cap5610 python=3.11 -y
conda activate cap5610
```

---

# 3. Detect Your CUDA Version

```bash
nvidia-smi
```

Note the **CUDA Version** in the top-right corner. Use the matching index below:

| CUDA Version | PyTorch Index URL |
|---|---|
| 11.8 | `https://download.pytorch.org/whl/cu118` |
| 12.1 | `https://download.pytorch.org/whl/cu121` |
| 12.4 | `https://download.pytorch.org/whl/cu124` |
| 12.6 | `https://download.pytorch.org/whl/cu126` |
| 13.x | use `cu126` (latest available) |

---

# 4. Install Project Dependencies

Install numpy first (from PyPI, not the torch index):

```bash
pip install numpy
```

Install PyTorch with your matching CUDA index (replace `<INDEX_URL>` with the URL from Step 3):

```bash
pip install torch==2.7.0 torchvision --index-url <INDEX_URL>
```

Install remaining project dependencies:

```bash
pip install -r requirements.txt --extra-index-url <INDEX_URL>
```

Install VideoMamba dependencies:

```bash
pip install timm einops
```

---

# 5. Install mamba-ssm

## Cloud GPU (recommended path)

On a cloud GPU instance, nvcc is available and you can build from source.
Use `--no-build-isolation` so pip uses the already-installed GPU torch instead of building in isolation with a CPU-only torch:

```bash
pip install mamba-ssm --no-build-isolation
```

## Local WSL (workaround)

Building from source in WSL is unreliable (excessive RAM, long build times). Use a pre-built wheel instead.

First check your Python version (`python --version`) and torch version (`python -c "import torch; print(torch.__version__)"`), then download the matching wheel from the mamba-ssm GitHub releases page:
- Match `cp311` for Python 3.11
- Match `torch2.7` for torch 2.7.x
- Use `cxx11abiTRUE` (standard for pip-installed PyTorch on Linux)

Install it:

```bash
pip install /path/to/mamba_ssm-*.whl
```

If you get `libcudart.so.11.0: cannot open shared object file`, install the CUDA 11 runtime:

```bash
pip install nvidia-cuda-runtime-cu11
```

---

# 6. Fix VideoMamba `__init__.py`

Replace the file at:

```
models/videomamba/models/__init__.py
```

With:

```python
# Minimal init for class project
```

This prevents unnecessary imports (CLIP, pretraining models, etc.)

---

# 7. Verify Imports (Smoke Test)

Run from the project root:

```bash
python - <<'PY'
import sys
sys.path.insert(0, ".")

mods = ["videomamba", "modeling_finetune", "deit"]
for m in mods:
    try:
        exec(f"from models.videomamba.models import {m}")
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: FAIL -> {e}")
PY
```

Expected outcome:

```
videomamba: OK
modeling_finetune: OK
deit: OK
```

---

# 8. Dataset Setup

Install the HuggingFace download library, then run the setup script:

```bash
pip install huggingface_hub
python scripts/download_dataset.py
```

This will:
1. Copy annotation files from `labels/` into `data/something-something-v2/annotations/` with the correct filenames
2. Download ~20GB of videos from HuggingFace (`HuggingFaceM4/something-something-v2`)

Expected structure after running:

```
data/something-something-v2/
├── 1.webm
├── 2.webm
├── ...
└── annotations/
    ├── something-something-v2-train.json
    ├── something-something-v2-validation.json
    ├── something-something-v2-test.json
    └── something-something-v2-labels.json
```

⚠️ On Vast.ai, make sure your instance has at least 80GB disk space before downloading.

---

# 9. Verify Data Loader

```bash
python - <<'PY'
from shared import get_dataloader

loader = get_dataloader(split="train", batch_size=2, num_frames=16)

for batch in loader:
    if batch is None:
        continue
    frames, labels = batch
    print(frames.shape)
    print(labels.shape)
    break
PY
```

Expected:

```
torch.Size([2, 16, 3, 224, 224])
torch.Size([2])
```

---

# 10. Next Step (Model Integration)

After setup is complete:

* Identify VideoMamba constructor (tiny or small)
* Integrate into your `train.py`
* Start with:

  * batch size: 2
  * frames: 16
  * resolution: 224

Then scale up based on VRAM

---

# 11. Recommended First Run Config

Running on A100 PCIE (40GB VRAM):

* Model: VideoMamba-Ti
* Frames: 16
* Resolution: 224
* Epochs: 30
* Mixed precision: enable `torch.cuda.amp` — cuts VRAM ~50%, letting you fit larger batches

**Batch size strategy — maximize to save money:**
Start high and binary search down until stable:

```
batch_size=64 → OOM? → try 32 → OOM? → try 16 → stable? → use that
```

With fp16 on an A100, expect to land around 32–64 for VideoMamba-Ti at 16 frames.
Use the largest batch that doesn't OOM. More data per step = fewer steps = less GPU time.

---

# 12. Important Notes

* Do NOT push model weights to repo
* Do NOT push dataset
* Only commit:

  * model code
  * training scripts
  * results JSON

---

# Done

You now have a clean, working environment ready for VideoMamba.

Next step: hook the model into your training loop.
