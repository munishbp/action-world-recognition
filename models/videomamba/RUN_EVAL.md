# VideoMamba SSv2-pretrained eval — Vast.ai runbook

Eval-only run of the SSv2-finetuned VideoMamba checkpoint on our SSv2 val
split. Mirrors the V-JEPA pattern: load released weights, run `validate()`,
write `results/VideoMambaSSv2Pretrained_results.json`.

## Pick a GPU

This is a one-pass eval over 24,777 clips. The bottleneck is video
decoding (CPU + I/O), not GPU compute.

| GPU            | Wall-clock estimate | Notes |
|----------------|--------------------:|-------|
| RTX 5090 (32G) | 25-40 min           | Same hardware used for other models in the benchmark |
| **A100 40/80G**| **20-30 min**       | **Recommended.** Comfortable VRAM headroom for batch 32 + bf16 |
| H100 80G       | 18-25 min           | Marginal speedup over A100 since the run is decode-bound |

A100 40GB is the recommended target. The eval run is bounded by video
decoding throughput rather than GPU compute, so a faster GPU yields
diminishing returns; an A100 with sufficient dataloader workers keeps the
GPU saturated.

## Run (paste-and-go)

Assumes a fresh Vast.ai box with the repo + SSv2 dataset already mounted
the same way you ran VideoMamba training.

```bash
cd /workspace/action-world-recognition  # adjust to your repo mount

# 1. Install (skip if already set up from previous training run)
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126

# 2. Download SSv2-finetuned VideoMamba-Small @ 16 frames
#    (highest-accuracy public checkpoint that fits comfortably on A100 40GB)
bash models/videomamba/download_ssv2_weights.sh small 16

# 3. Run eval — about 20-30 min on A100, slightly less on H100
python models/videomamba/eval.py \
    --model small \
    --num-frames 16 \
    --ckpt models/videomamba/checkpoints/videomamba_s16_ssv2_f16_res224.pth \
    --batch-size 32 \
    --num-workers 8

# 4. Results land at:
ls results/VideoMambaSSv2Pretrained_results.json \
   results/VideoMambaSSv2Pretrained_confusion_matrix.npy
```

## Expected numbers

Paper reports for `videomamba_s16_ssv2_f16_res224`: **67.6% top-1** on
SSv2 val.

V-JEPA (similar eval-only run) lost ~5 points to environmental drift on our
setup (FP16 vs BF16, decode patches). Expect VideoMamba to land in the
**62-67%** range. If you want to maximize the chance of matching the paper
number exactly, also try `--num-workers 4` and remove the bf16 autocast
(`--no-fp16`) — but this is slower.

## If the download URL 404s

The HuggingFace mirror at `OpenGVLab/VideoMamba` is the canonical source.
If filenames have changed, find the current name in:
- https://github.com/OpenGVLab/VideoMamba/blob/main/videomamba/video_sm/MODEL_ZOO.md

Then either edit `download_ssv2_weights.sh` or just `wget` directly into
`models/videomamba/checkpoints/`.

## What to put in RESULTS.md

Add a row alongside V-JEPA in the "SOTA baseline" / eval-only category:

| Model | Type | Owner | Top-1 | Top-5 | F1 | Params | Trainable |
|-------|------|-------|------:|------:|---:|-------:|----------:|
| **VideoMamba-S (SSv2 ckpt)** | **SSM (eval only)** | Kenneth | 0.XXXX | 0.XXXX | 0.XXXX | 26M | 0 |

In the Training Efficiency table, mark Training Time = "N/A (eval only)"
like V-JEPA, and put the Peak VRAM the eval reported.
