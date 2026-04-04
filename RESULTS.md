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
| ST-GCN | GNN | Munish | | | | 3.1M | 3.1M |
| PredRNN | World Model | Munish | | | | | |
| Qwen3.5-9B | VLM (QLoRA) | Munish | | | | | |

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
| ST-GCN | | | 16 | N/A | 64 | 50 | |
| PredRNN | | | | 224 | | | |
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
- Fine-tuning strategy: full training, SGD, LR 0.1 decayed at epochs 30/40
- Keypoint extraction: MediaPipe PoseLandmarker, 33 joints, 16 frames/video
- Detection rate (val): 55.6% (expected -- SSv2 often shows only hands, no full body)
- Input: skeleton keypoints (x, y, visibility, dx, dy) -- no RGB
- Best val epoch:
- What worked:
- What didn't:
- Failure modes:

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
