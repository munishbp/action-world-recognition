# CAP 5610 - Video Action Recognition Benchmark

Benchmarking 10 video understanding models on [Something-Something V2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset) (174 action classes, ~220K video clips).

## Models

| Model | Type | VRAM Est. |
|-------|------|-----------|
| TSM | CNN | ~6 GB |
| R(2+1)D | CNN | ~8 GB |
| SlowFast | CNN | ~10 GB |
| TimeSformer | Transformer | ~16 GB |
| VideoMAE | Transformer | ~18 GB |
| VideoMamba | SSM | ~10 GB |
| CNN+ConvLSTM | CNN+RNN | ~8 GB |
| ST-GCN | GNN | ~4 GB |
| PredRNN | World Model | ~12 GB |
| Qwen3.5-9B | VLM (QLoRA) | ~24 GB |
| V-JEPA (ViT-L) | SOTA Baseline (eval only) | ~10 GB |

## Setup

```bash
# Create environment
conda create -n cap5610 python=3.11 -y
conda activate cap5610

# Install dependencies (CUDA 12.6)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# For exact reproducibility, use the lock file instead
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

## Dataset

Download Something-Something V2 from [Qualcomm](https://www.qualcomm.com/developer/software/something-something-v-2-dataset) and place it under `data/`:

```
data/something-something-v2/
├── 1.webm
├── 2.webm
├── ...                          # ~220K video clips (WebM, VP9, 240p)
└── annotations/
    ├── something-something-v2-labels.json        # 174 label names -> indices
    ├── something-something-v2-train.json          # 168,913 entries
    ├── something-something-v2-validation.json     # 24,777 entries
    └── something-something-v2-test.json           # 27,157 entries (no labels)
```

Each annotation entry:
```json
{"id": "41775", "label": "Moving part of something", "template": "Moving part of [something]", "placeholders": ["drawer"]}
```

## Shared Pipeline

### Data Loading

```python
from shared import get_dataloader

train_loader = get_dataloader(split="train", batch_size=16, num_frames=8)
val_loader = get_dataloader(split="val", batch_size=32, num_frames=8)

for batch in train_loader:
    if batch is None:
        continue  # skipped corrupt videos (extremely rare, see note below)
    frames, labels = batch
    # frames: (B, T, C, H, W) float32, ImageNet-normalized
    # labels: (B,) int64
    ...
```

**Parameters:**
- `num_frames` — frames to sample per video (default 8, use 16 for transformers)
- `frame_size` — spatial resolution (default 224)
- `num_workers` — dataloader workers (default 4)
- `transform` — pass a custom `torchvision.transforms.Compose` to override **all** default transforms

### What the pipeline does for you

**Video loading** — Tries decord first (faster), automatically falls back to OpenCV if decord chokes on a video. If both fail, the video is skipped with a warning instead of crashing your training run. You don't need to handle this.

**Augmentation (train split only):**
- `RandomResizedCrop(224, scale=0.6-1.0)` — random spatial crops
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)` — color augmentation
- ImageNet normalization

**Val/test transforms:**
- `Resize(256)` + `CenterCrop(224)` — deterministic, no randomness
- ImageNet normalization

**Important: NO horizontal flip.** SSv2 is direction-sensitive. "Pushing something from left to right" and "Pushing something from right to left" are different classes. If you add your own transforms, do not flip.

### If your model needs something different

The shared pipeline returns `(B, T, C, H, W)`. If your model needs a different format:

- **R(2+1)D** needs `(B, C, T, H, W)` — just add `frames = frames.permute(0, 2, 1, 3, 4)` in your training loop
- **SlowFast** needs dual pathways — sample your own slow/fast frames on top of the shared loader, or write a custom dataset that wraps the shared one
- **Qwen / VLMs** need text+vision — you'll need your own data loading for the text side

If you pass `transform=` to override, you get full control. The shared defaults are just there so you don't have to think about it for standard models.

### The `batch is None` thing

If every single video in a batch fails to decode, the batch comes back as `None`. This should basically never happen (every video in a batch of 16 would have to be corrupt), but guard against it:

```python
for batch in train_loader:
    if batch is None:
        continue
    frames, labels = batch
    # ...
```

One line. Just do it.

### Evaluation

From Python:
```python
from shared import evaluate_model, save_results

results = evaluate_model(
    all_logits,                    # (N, 174) logits or (N,) predicted class indices
    all_labels,                    # (N,) ground truth
    model_name="TSM",
    training_time_hours=10.5,
    peak_vram_gb=6.2,
    total_params=24_000_000,
    trainable_params=24_000_000,
)
save_results(results, output_dir="results")
# -> results/TSM_results.json
# -> results/TSM_confusion_matrix.npy
```

From CLI:
```bash
python -m shared.evaluate \
    --predictions results/TSM_logits.npy \
    --labels results/TSM_labels.npy \
    --model-name TSM \
    --training-time 10.5 \
    --peak-vram 6.2 \
    --total-params 24000000 \
    --trainable-params 24000000
```

**Output JSON format:**
```json
{
  "model_name": "TSM",
  "top1_acc": 0.4523,
  "top5_acc": 0.7681,
  "f1_weighted": 0.4489,
  "per_class_acc": {"0": 0.52, "1": 0.38, "...": "..."},
  "training_time_hours": 10.5,
  "peak_vram_gb": 6.2,
  "total_params": 24000000,
  "trainable_params": 24000000
}
```

## Project Structure

```
.
├── shared/                  # Shared infrastructure (everyone imports this)
│   ├── dataset.py           # SSv2 Dataset + DataLoader
│   └── evaluate.py          # Metrics + result serialization
├── models/
│   ├── tsm/
│   ├── r2plus1d/
│   ├── slowfast/
│   ├── timesformer/
│   ├── videomae/
│   ├── videomamba/
│   ├── cnn_convlstm/
│   ├── stgcn/
│   ├── predrnn/
│   └── qwen/
├── results/                 # Eval outputs (*_results.json + *_confusion_matrix.npy)
├── configs/
│   └── default.yaml         # Shared defaults
├── requirements.txt         # Minimum dependency versions
└── requirements-lock.txt    # Pinned versions for reproducibility
```

## Team

- **Munish Persaud** — [munishbp.com](https://munishbp.com)
- **Ayaan** —
- **Aiden** —
- **Kenneth** —
- **Arthur** —
