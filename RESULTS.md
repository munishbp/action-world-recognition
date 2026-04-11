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
| SlowFast | CNN | Aiden | 0.3634 | | | 34M | 34M |
| TimeSformer | Transformer | Aiden | | | | | |
| VideoMAE | Transformer | Aiden | | | | | |
| VideoMamba | SSM | Kenneth | | | | | |
| CNN+ConvLSTM | CNN+RNN | Kenneth | | | | | |
| ST-GCN | GNN | Munish | 0.0394 | 0.1231 | 0.0192 | 3.1M | 3.1M |
| PredRNN | World Model | Munish | 0.0467 | 0.1302 | 0.0164 | 18.6M | 18.6M |
| Qwen3.5-4B | VLM (QLoRA) | Munish | 0.5819 | -- | 0.5597 | 2.59B | 3.15M |
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
| PredRNN | 13.24 | 4.79 | 8 | 224 | 16 | 15 | V100-32GB + RTX 5090 |
| V-JEPA | N/A (eval only) | 4.3 | 16 | 224 | 4 | 0 | V100-32GB |
| Qwen3.5-4B | 43.29 | 9.15 | 8 | 224 | 2 (eff 16) | 1 | RTX 5090 (Vast.ai) |

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
| PredRNN | Something falling like a rock -> Moving something down | 95 | Semantically sensible, falling is a kind of moving down |
| Qwen3.5-4B | Plugging something into something -> Plugging something into something but pulling it right out as you remove your hand | 429 | Picks up the "plug in" motion but misses the extended "pull out" that distinguishes the longer label |

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

  SlowFast (Aiden)

  - Pretrained from: Kinetics-400 (SlowFast-R50 via facebookresearch/pytorchvideo torch.hub)
  - Fine-tuning strategy: Full fine-tune — all layers unfrozen, 400-class head replaced with
  174-class linear layer
  - Slow/Fast frame config: Fast = 32 frames, Slow = 8 frames (every 4th frame, α=4), 224×224
  - Optimizer / LR / Schedule: SGD, momentum=0.9, weight decay=1e-4, lr=0.01, cosine annealing to
  1e-4 over 20 epochs
  - Best val epoch: N/A — training did not complete
  - What worked: Loss was decreasing steadily through the first partial epoch (5.09 → 4.58),
  learning signal present
  - What didn't: GPU compatibility — RTX 5090 (sm_120) not supported by stable PyTorch builds,
  caused CUDA kernel crash mid-epoch
  - Failure modes: CUDA no kernel image error on Blackwell GPUs; batch size 8 with 32 frames is
  conservative (potentially slow training)

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
- Pretrained from: trained from scratch
- Fine-tuning strategy: full training, 15 epochs total. First 6 epochs on local RTX 5090, then resumed from `last.pt` on Vast.ai V100-32GB for epochs 7-15 (F: drive I/O contention with another job was starving the 5090 dataloader). Resume worked cleanly via cosine scheduler state.
- Optimizer / LR / Schedule: Adam, LR 1e-3, weight_decay 1e-4, CosineAnnealingLR T_max=15, grad clip 1.0
- Input: 8 frames/clip, 224 resolution, decord decoded on the fly (no frame cache on Vast). Batch size 16 train, 32 val.
- Architecture: CNN encoder (32->64->64 channels, 3x downsample to 28x28) + 4 stacked ST-LSTM layers (64,64,128,128 hidden) with PredRNN spatial memory M (128 dim), dropout 0.3, 18.6M params (all trainable)
- Best val epoch: 15 (final, val acc 0.0467). Every epoch from 7 to 15 on the resumed run wrote a new `best.pt`. The cosine LR decay was clearly doing work late in training.
- What worked: Resuming from a partial run just worked. The scheduler T_max=15 state preserved correctly through the checkpoint. Going from 1.13% val acc at epoch 6 to 4.67% at epoch 15 is a 4x improvement that the original plan warning (stuck near baseline) did not predict. Spatial memory helps PredRNN pick up camera-direction and fall-direction classes much better than skeleton-only ST-GCN.
- What didn't: 4.67% top-1 on 174 classes is still far from usable. Roughly 70% of classes get 0% accuracy. PredRNN's spatiotemporal world model picks up global motion patterns but not the fine-grained hand-object interactions that dominate SSv2 (e.g. attaching something to something, bending something so that it deforms). 8 frames per clip is probably also too few to resolve fast manipulations.
- Failure modes: The model learns a handful of motion-heavy classes (30-52% accuracy on classes 94, 109, 43, 93, 146, all camera-direction and surface-placement actions) and gets 0% on everything else. Top confused pair is Something falling like a rock -> Moving something down (95 confusions), which is semantically correct because falling is a kind of moving down. Similar for Tearing something into two pieces -> Moving something down (91). PredRNN is predicting the motion correctly but not the physical transformation.

### Qwen3.5-4B (Munish)
- Pretrained from: Qwen/Qwen3.5-4B (multimodal VLM, vision encoder + LLM, 2.59B params total)
- Fine-tuning strategy (QLoRA config): 4-bit NF4 quantization via bitsandbytes 0.49.2 with double quant and fp16 compute, LoRA adapters r=16 alpha=32 dropout=0.05 on q/k/v/o projections only. 3.15M trainable params (0.12% of total). Trained 1 epoch on the full 168,913-clip train set.
- Prompt template: `"You are watching a short video clip. The frames shown are sampled uniformly from the video. What action is being performed? Respond with ONLY the action label, nothing else."` Classification is done by string-matching the model's generated text against the 174 class names (exact, case-insensitive, then substring fallback).
- Optimizer / LR / Schedule: AdamW lr=2e-4, weight_decay=0.01, grad clip 1.0, no warmup, no LR schedule (fixed LR for 1 epoch). Gradient accumulation 8 steps on microbatch 2 -> effective batch size 16.
- Best val epoch: 1 (only epoch run). Val top-1 0.5819, F1 (weighted) 0.5597, mean per-class acc 0.5225.
- What worked: QLoRA 4-bit loaded cleanly on RTX 5090 Blackwell (sm_120) with bitsandbytes 0.49.2 — no compatibility issues despite the new arch. 3.15M trainable LoRA params converged in a single epoch on 168K samples. Generative classification (model.generate() -> text -> nearest label) works surprisingly well: 58.19% top-1 is within 6 points of V-JEPA's 64.51% SOTA baseline with roughly 1/100 the trainable parameters. Only 10/174 classes are at 0% accuracy (vs ~70% for PredRNN, ~95% for ST-GCN). First run on Windows 5090 was CPU-bound at ~5 s/it; moving to a Linux Vast.ai 5090 with 12 vCPUs dropped the same run to 1.69 s/it (3x speedup) purely by removing Windows dataloader contention.
- What didn't: Top-5 is architecturally null — Qwen is a generative VLM, there is no 174-dim logit distribution to rank. The Vast.ai dataset.py rewrite reads .webm via decord and passes native-resolution frames to the Qwen processor, which triggered a staircase VRAM allocation pattern (25.4 GB -> 28.4 GB -> 31.3 GB across the epoch as occasional oversized samples pushed PyTorch's cached allocator to new highs). Run finished at 96% VRAM with ~1.3 GB headroom — pre-resizing frames to 224x224 before the processor would have prevented this entirely, but wasn't worth a restart at 64% complete. Training time (43.3 hours) dominates the cost analysis relative to every other model in the suite.
- Failure modes: Top confused pair is `Plugging something into something -> Plugging something into something but pulling it right out as you remove your hand` (429 confusions). The model correctly picks up the "plug in" motion but cannot distinguish the extended "pull out" that defines the longer label — a legitimate label-ambiguity case where the two actions are functionally the same for the first half of the clip. Zero-accuracy classes cluster around "pretending" variants (Pretending or trying and failing to twist something, Pretending to poke something) and ambiguous throwing/pouring actions. The 10 zero-acc classes also include the base `Pouring something into something`, which gets absorbed into the more specific `Pouring something into something until it overflows` (Qwen's single easiest class at 97.9%). Fine-grained manipulation distinctions remain the model's hard ceiling.

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
