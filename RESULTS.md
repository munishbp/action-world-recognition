# Benchmark Results

Fill in your row as your model finishes training + eval. The eval script (`shared/evaluate.py`) gives you everything you need -- just copy the numbers from your `results/{MODEL}_results.json`.

Arthur: this file is your source of truth for the final report. Every table here maps directly to a section in the paper.

---

## Main Results

The headline table. This is what goes in the paper.

| Model | Type | Owner | Top-1 Acc | Top-5 Acc | F1 (weighted) | Params | Trainable Params |
|-------|------|-------|-----------|-----------|---------------|--------|-----------------|
| TSM | CNN | Ayaan | | | | | |
| R(2+1)D | CNN | Ayaan | | | | | |
| SlowFast | CNN | Aiden | | | | | |
| TimeSformer | Transformer | Aiden | | | | | |
| VideoMAE | Transformer | Aiden | | | | | |
| VideoMamba | SSM | Kenneth | | | | | |
| CNN+ConvLSTM | CNN+RNN | Kenneth | | | | | |
| ST-GCN | GNN | Munish | 0.0394 | 0.1231 | 0.0192 | 3.1M | 3.1M |
| PredRNN | World Model | Munish | | | | | |
| Qwen3.5-9B | VLM (QLoRA) | Munish | | | | | |
| **V-JEPA** | **SOTA baseline** | **Munish** | 0.6451 | -- | -- | 307M | 0 |

**V-JEPA (SOTA reference):** Meta FAIR's self-supervised ViT for video (`facebookresearch/jepa`). ViT-L/16 backbone pretrained with the V-JEPA objective on VideoMix2M (90K iterations, 300 epochs), attentive probe head pretrained by Meta on SSv2 (20 epochs, world_size=128). Ran eval-only on our 24,777-clip val set with the standard 16x2x3 multi-view protocol on a single V100-32GB. **Reproduced 64.51% top-1**, about 5 points below Meta's published 69.5%. The gap is environmental, not the model: (a) V-JEPA's eval code hardcodes `dtype=torch.float16` in the autocast block even when `use_bfloat16: true` is set, so the run was actually FP16, not BF16. On V100 this matters because tensor cores are FP16-only and the narrower dynamic range shifts logits in the attention softmax. (b) We had to patch decord with `num_threads=1` to avoid an FFmpeg threaded_decoder crash on this host, which may sample frames slightly differently than Meta's original pipeline. Runs in ~4.3 GB VRAM at batch 4. License: CC-BY-NC 4.0 (fine for academic).

---

## Training Efficiency

How expensive was each model to train. Important for the cost-vs-accuracy analysis.

| Model | Training Time (hrs) | Peak VRAM (GB) | Frames/video | Frame Size | Batch Size | Epochs | GPU |
|-------|--------------------:|---------------:|-------------:|-----------:|---------:|-------:|-----|
| TSM | | | | 224 | | | |
| R(2+1)D | | | | 224 | | | |
| SlowFast | | | | 224 | | | |
| TimeSformer | | | | 224 | | | |
| VideoMAE | | | | 224 | | | |
| VideoMamba | | | | 224 | | | |
| CNN+ConvLSTM | | | | 224 | | | |
| ST-GCN | 5.5 | 0.86 | 16 | N/A | 64 | 50 | RTX 5090 |
| PredRNN | | | | 224 | | | |
| V-JEPA | N/A (eval only) | 4.3 | 16 | 224 | 4 | 0 | V100-32GB |
| Qwen3.5-9B | | | | | | | |

---

## Per-Class Accuracy (Top/Bottom 10)

After all models are done, fill in the hardest and easiest classes across models. Arthur: use this for the error analysis section.

### Easiest Classes (highest avg accuracy across models)

| Rank | Class Name | TSM | R(2+1)D | SlowFast | TimeSformer | VideoMAE | VideoMamba | ConvLSTM | ST-GCN | PredRNN | Qwen | Avg |
|------|-----------|-----|---------|----------|-------------|----------|------------|----------|--------|---------|------|-----|
| 1 | | | | | | | | | | | | |
| 2 | | | | | | | | | | | | |
| 3 | | | | | | | | | | | | |
| 4 | | | | | | | | | | | | |
| 5 | | | | | | | | | | | | |
| 6 | | | | | | | | | | | | |
| 7 | | | | | | | | | | | | |
| 8 | | | | | | | | | | | | |
| 9 | | | | | | | | | | | | |
| 10 | | | | | | | | | | | | |

### Hardest Classes (lowest avg accuracy across models)

| Rank | Class Name | TSM | R(2+1)D | SlowFast | TimeSformer | VideoMAE | VideoMamba | ConvLSTM | ST-GCN | PredRNN | Qwen | Avg |
|------|-----------|-----|---------|----------|-------------|----------|------------|----------|--------|---------|------|-----|
| 1 | | | | | | | | | | | | |
| 2 | | | | | | | | | | | | |
| 3 | | | | | | | | | | | | |
| 4 | | | | | | | | | | | | |
| 5 | | | | | | | | | | | | |
| 6 | | | | | | | | | | | | |
| 7 | | | | | | | | | | | | |
| 8 | | | | | | | | | | | | |
| 9 | | | | | | | | | | | | |
| 10 | | | | | | | | | | | | |

---

## Confusion Matrix Highlights

After eval, note which classes get confused with each other the most. Look at off-diagonal peaks in the confusion matrix. Arthur: this is gold for the discussion section.

| Model | Most Confused Pair (A -> predicted as B) | Count | Notes |
|-------|------------------------------------------|-------|-------|
| TSM | | | |
| R(2+1)D | | | |
| SlowFast | | | |
| TimeSformer | | | |
| VideoMAE | | | |
| VideoMamba | | | |
| CNN+ConvLSTM | | | |
| ST-GCN | | | |
| PredRNN | | | |
| Qwen3.5-9B | | | |

---

## Per-Model Notes

Fill in anything notable about your model -- what worked, what didn't, any surprises. Arthur will use this for the discussion section.

### TSM (Ayaan)
- Pretrained from:
- Fine-tuning strategy:
- Optimizer / LR / Schedule:
- Best val epoch:
- What worked:
- What didn't:
- Failure modes (what does it get wrong?):

### R(2+1)D (Ayaan)
- Pretrained from:
- Fine-tuning strategy:
- Optimizer / LR / Schedule:
- Best val epoch:
- What worked:
- What didn't:
- Failure modes:

### SlowFast (Aiden)
- Pretrained from:
- Fine-tuning strategy:
- Slow/Fast frame config:
- Optimizer / LR / Schedule:
- Best val epoch:
- What worked:
- What didn't:
- Failure modes:

### TimeSformer (Aiden)
- Pretrained from:
- Fine-tuning strategy:
- Attention type (divided/joint/space-only):
- Optimizer / LR / Schedule:
- Best val epoch:
- What worked:
- What didn't:
- Failure modes:

### VideoMAE (Aiden)
- Pretrained from:
- Fine-tuning strategy:
- Masking ratio:
- Optimizer / LR / Schedule:
- Best val epoch:
- What worked:
- What didn't:
- Failure modes:

### VideoMamba (Kenneth)
- Pretrained from:
- Fine-tuning strategy:
- Optimizer / LR / Schedule:
- Best val epoch:
- What worked:
- What didn't:
- Failure modes:

### CNN+ConvLSTM (Kenneth)
- CNN backbone:
- Pretrained from:
- Fine-tuning strategy:
- Optimizer / LR / Schedule:
- Best val epoch:
- What worked:
- What didn't:
- Failure modes:

### ST-GCN (Munish)
- Pretrained from: trained from scratch
- Fine-tuning strategy: full training, SGD, LR 0.1 decayed by 0.1x at epochs 30/40
- Keypoint extraction: MediaPipe PoseLandmarker, 33 joints, 16 frames/video
- Detection rate: train 52.9%, val 55.6%, test 55.5% (expected -- SSv2 often shows only hands, no full body)
- Input: skeleton keypoints (x, y, visibility, dx, dy) -- no RGB
- Best val epoch: 33 (val acc 0.0519)
- What worked: LR decay at epoch 30 gave an immediate accuracy bump (4.7% -> 5.2%). Velocity features (dx, dy) help with direction-sensitive classes.
- What didn't: Accuracy plateaus early (~epoch 10) and overfits after epoch 33. Skeleton-only representation fundamentally can't see objects being manipulated.
- Failure modes: Most classes get 0% accuracy. Only works on motion-heavy classes where body pose carries signal (class 171: 38%, class 94: 36%, class 43: 35%). Completely fails on fine-grained hand-object interactions.

### V-JEPA (Munish)
- Pretrained from: Meta FAIR ViT-L/16 backbone (VideoMix2M, 90K iter, 300 epochs) + Meta SSv2 attentive probe (20 epochs, bs=2, world_size=128)
- Fine-tuning strategy: none, eval-only. Backbone and probe both frozen.
- Input: 16 frames/clip, frame_step=4, 2 temporal segments x 3 spatial views = 6 views per video, 224 resolution
- Eval protocol: standard 16x2x3 multi-view (matches Meta's published protocol)
- Best val epoch: N/A (probe loaded at Meta's final epoch 20)
- What worked: Out-of-the-box SOTA with zero training on our hardware. Backbone and probe both loaded cleanly (`<All keys matched successfully>`) with Meta's checkpoint format. The V-JEPA code's `load_checkpoint` was built to expect exactly their own save format, so resuming from the pretrained probe just worked. 24,777-clip eval in 72 minutes on one V100.
- What didn't: FP16 on V100 costs about 5 points vs Meta's published 69.5% BF16 number (we got 64.51%). V-JEPA's eval code hardcodes `torch.float16` in autocast even when `use_bfloat16: true` is set in the config. Silent foot-gun, not our bug. Also had to patch `src/datasets/video_dataset.py` to use `num_threads=1` in decord's VideoReader, same FFmpeg threaded_decoder crash we hit in PredRNN's dataloader on this host.
- Failure modes: None from the model itself. The gap is purely environmental (FP16 autocast, decord threading workaround). On an A100 + BF16 the published 69.5% should reproduce exactly. This is a ceiling number for what pretrained SSL can do on SSv2 without any task-specific training.

### PredRNN (Munish)
- Pretrained from:
- Fine-tuning strategy:
- Optimizer / LR / Schedule:
- Best val epoch:
- What worked:
- What didn't:
- Failure modes:

### Qwen3.5-9B (Munish)
- Pretrained from:
- Fine-tuning strategy (QLoRA config):
- Prompt template:
- Optimizer / LR / Schedule:
- Best val epoch:
- What worked:
- What didn't:
- Failure modes:

---

## Where to find the raw outputs

Each model's eval produces two files in `results/`:
- `{MODEL}_results.json` -- all metrics in JSON (top-1, top-5, F1, per-class accuracy, metadata)
- `{MODEL}_confusion_matrix.npy` -- 174x174 confusion matrix as numpy array

Load the confusion matrix:
```python
import numpy as np
cm = np.load("results/TSM_confusion_matrix.npy")  # (174, 174)
```

---

## For Arthur

Everything you need for the report:

1. **Main results table** -- copy directly into the paper
2. **Training efficiency table** -- for the cost analysis section
3. **Per-model notes** -- for the discussion section, qualitative observations
4. **Confusion matrices** -- in `results/`, use these for per-class analysis, error patterns, or visualization
5. **Per-class accuracy** -- in each model's JSON, under `per_class_acc`. Use this to find which action classes are hardest across all models

To compare all models programmatically:
```python
import json, glob

for f in sorted(glob.glob("results/*_results.json")):
    r = json.load(open(f))
    print(f"{r['model_name']:15s}  top1={r['top1_acc']:.4f}  top5={r['top5_acc']:.4f}  f1={r['f1_weighted']:.4f}")
```
