# Benchmark Results

Fill in your row as your model finishes training + eval. The eval script (`shared/evaluate.py`) gives you everything you need -- just copy the numbers from your `results/{MODEL}_results.json`.

Arthur: this file is your source of truth for the final report. Every table here maps directly to a section in the paper.

---

## Main Results

The headline table. This is what goes in the paper.

| Model | Type | Owner | Top-1 Acc | Top-5 Acc | F1 (weighted) | Params | Trainable Params |
|-------|------|-------|-----------|-----------|---------------|--------|-----------------|
| TSM | CNN | Ayaan |0.06265 |0.15305 |0.06817 | 23.8M|23.8M |
| R(2+1)D | CNN | Ayaan | 0.4636|0.7484 |0.4611 |31.3M |31.3M |
| SlowFast | CNN | Arthur | 0.3634 | | | 34M | 34M |
| TimeSformer | Transformer | Aiden | 0.3441 | 0.6934 | 0.5734 | 121M | 121M |
| VideoMAE | Transformer | Aiden | 0.1692 | 0.3984 | 0.1584 | 86.4M | 86.4M |
| VideoMamba (K400 finetuned, post-bugfix) | SSM | Kenneth | 0.5391 | 0.8273 | 0.5290 | 26.0M | 26.0M |
| VideoMamba (from-scratch, pre-bugfix) | SSM | Kenneth | 0.0145 | 0.0613 | 0.0004 | 6.3M | 6.3M |
| CNN+ConvLSTM | CNN+RNN | Kenneth | 0.2701 | 0.5314 | 0.2472 | 18.3M | 18.3M |
| ST-GCN | GNN | Arthur | 0.0394 | 0.1231 | 0.0192 | 3.1M | 3.1M |
| PredRNN | World Model | Munish | 0.0467 | 0.1302 | 0.0164 | 18.6M | 18.6M |
| Qwen3.5-4B | VLM (QLoRA) | Munish | 0.5819 | -- | 0.5597 | 2.59B | 3.15M |
| **V-JEPA** | **SOTA baseline** | **Munish** | 0.6451 | -- | -- | 307M | 0 |

**V-JEPA (SOTA reference):** Meta FAIR's self-supervised ViT for video (`facebookresearch/jepa`). ViT-L/16 backbone pretrained with the V-JEPA objective on VideoMix2M (90K iterations, 300 epochs), attentive probe head pretrained by Meta on SSv2 (20 epochs, world_size=128). Ran eval-only on our 24,777-clip val set with the standard 16x2x3 multi-view protocol on a single V100-32GB. **Reproduced 64.51% top-1**, about 5 points below Meta's published 69.5%. The gap is environmental, not the model: (a) V-JEPA's eval code hardcodes `dtype=torch.float16` in the autocast block even when `use_bfloat16: true` is set, so the run was actually FP16, not BF16. On V100 this matters because tensor cores are FP16-only and the narrower dynamic range shifts logits in the attention softmax. (b) We had to patch decord with `num_threads=1` to avoid an FFmpeg threaded_decoder crash on this host, which may sample frames slightly differently than Meta's original pipeline. Runs in ~4.3 GB VRAM at batch 4. License: CC-BY-NC 4.0 (fine for academic).

---

## Training Efficiency

How expensive was each model to train. Important for the cost-vs-accuracy analysis.

| Model | Training Time (hrs) | Peak VRAM (GB) | Frames/video | Frame Size | Batch Size | Epochs | GPU |
|-------|--------------------:|---------------:|-------------:|-----------:|---------:|-------:|-----|
| TSM |13.17 |6.39 | 8| 224 |8 |30 |RTX 5090 |
| R(2+1)D |105.22 |12.53 |8 | 224 |8 |30 |RTX 5090 |
| SlowFast | 48 | 16 | 8 | 224 | 13 | RTX 5090 |
| TimeSformer | 35 | 16 | 8 | 224 | 8 | 15 | RTX 5090 |
| VideoMAE | 19.17 | 40.56 | 16 | 224 | 64 | 11 (of 15) | RTX Pro 6000 Blackwell 96GB (Vast.ai) |
| VideoMamba (K400 finetuned) | 7.0 | ~24 | 16 | 224 | 16 | 2 (of planned 15) | A100 |
| VideoMamba (from-scratch, pre-bugfix) | 4.37 | 38.02 | 16 | 224 | 16 | 10 (of 30) | unknown (likely A100, per SETUP.MD) |
| CNN+ConvLSTM | 9.63 | 6.18 | 8 | 224 | 32 | 15 | A100 SXM4 |
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
| TSM |Moving something and something so they pass each other -> Something colliding with something and both are being deflected | 7| Similar actions however missing the context of object is colliding and when they are passing one another|
| R(2+1)D |Removing something, revealing something behind -> Moving part of something|54|Similar-looking motion hides different interaction semantics when the model misses how objects relate to each other|
| SlowFast | | | |
| TimeSformer | | | |
| VideoMAE | Tearing something just a little bit -> Tearing something into two pieces | 57 | Both involve the same tearing motion; the distinction is degree, not kinematics |
| VideoMamba | Tearing something just a little bit -> Tearing something into two pieces | 76 | Same most-confused pair as VideoMAE and CNN+ConvLSTM; the tearing motion is identical at clip start, the distinction is degree-of-completion (a state-classification problem, not a motion one) — a recurring weak spot across architectures on SSv2 |
| CNN+ConvLSTM | Tearing something just a little bit -> Tearing something into two pieces | 70 | Same as VideoMAE/VideoMamba: model sees the tearing motion but cannot disambiguate "partial" from "complete" because the ConvLSTM compresses the temporal trajectory before the end-state is visible |
| ST-GCN | | | |
| PredRNN | Something falling like a rock -> Moving something down | 95 | Semantically sensible, falling is a kind of moving down |
| Qwen3.5-4B | Plugging something into something -> Plugging something into something but pulling it right out as you remove your hand | 429 | Picks up the "plug in" motion but misses the extended "pull out" that distinguishes the longer label |

---

## Per-Model Notes

Fill in anything notable about your model -- what worked, what didn't, any surprises. Arthur will use this for the discussion section.

### TSM (Ayaan)
- Pretrained from: ResNet50 with the ImageNet dataset, default weights
- Fine-tuning strategy: Start from ImageNet-pretrained ResNet-50, inject temporal shift in each bottleneck, swap the head for 174 classes, and update the whole network
- Optimizer / LR / Schedule: SGD, momentum=0.9, weight decay = 1e-4, Peak Learning Rate 0.02, Cosine Annealing Learning to 2.5e-04 over 30 epochs
- Best val epoch: Epoch 30, val accuracy of 0.0626
- What worked: Training accuracy went to 96% and loss drops steadily so model fits the training set
- What didn't: Validation accuracy was between 1.7-6.3% very spiky, huge gap between training and val, maybe some overfitting and memorization
- Failure modes (what does it get wrong?): Most Validation classes results may be random, errors are not class specific

### R(2+1)D (Ayaan)
- Pretrained from: R2Plus1D_18 with defualt weights
- Fine-tuning strategy: Start from Torchvision’s pretrained R(2+1)D-18, replace the final FC with 174 outputs, and fine-tune the full 3D backbone and head on multi-frame clips
- Optimizer / LR / Schedule: SGD, momentum=0.9, weight decay = 1e-4, Peak Learning Rate 0.01, Cosine Annealing Learning to 1.3e-04 over 30 epochs
- Best val epoch: Epoch 30, val accuracy of 0.4635
- What worked: Smooth train and val improvement through ~epoch 20–25; val reaches ~46% with 168,912 train samples and 24,777 val samples used
- What didn't:Late epochs: train acc ~97% vs val ~46% → overfitting; gains after ~epoch 21–25 are small while val loss creeps up (~2.55 → ~2.85)
- Failure modes: At ~46% val on 174 classes you still expect verb / motion confusions and hard tail classes

### SlowFast (Arthur)
- Pretrained from:
- Fine-tuning strategy:
- Slow/Fast frame config:
- Optimizer / LR / Schedule:
- Best val epoch:
- What worked:
- What didn't:
- Failure modes:

  SlowFast (Arthur)

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

### TimeSformer (Aiden)
- Pretrained from: `facebook/timesformer-base-finetuned-k400` (Kinetics-400 fine-tuned, 400-class head replaced with fresh 174-class linear via `ignore_mismatched_sizes=True`)
- Architecture: divided space-time attention (separate temporal + spatial attention blocks), 8 frames, 224×224, 121M params
- Fine-tuning strategy: full fine-tune — all layers unfrozen
- Optimizer / LR / Schedule: AdamW, lr=1e-4, weight_decay=0.05, cosine annealing to 1e-6 over 15 epochs
- Mixed precision: bf16
- Top-1: 57%
- Best val epoch: 12
- What worked: K400 pretraining gave a strong initialization — 34.4% top-1 well above random (0.6% for 174 classes) and competitive with SlowFast. Top-5 of 69.3% shows the correct class lands in the model's top predictions most of the time, meaning it generally understands the action category but struggles with fine-grained disambiguation. Weighted F1 of 0.57 indicates balanced performance across classes rather than memorizing a few dominant ones. bf16 mixed precision was stable throughout all 15 epochs with no loss divergence.
- What didn't: 34.4% top-1 is well below the ~59% reported in the original paper. Three likely causes: (a) only 8 frames per clip limits temporal resolution for SSv2's fast hand-object manipulations — the model sees too few frames to resolve motion direction and speed reliably; (b) batch size 8 is small due to VRAM constraints, producing noisy gradient estimates that slow convergence; (c) 15 epochs on 168K training samples may not be enough for full convergence given the small batch. The divided space-time attention also processes temporal and spatial attention separately, which may miss joint space-time interactions that are critical for SSv2's direction-sensitive classes.
- Failure modes: SSv2 is dominated by fine-grained hand-object interactions where the distinction between classes is often subtle (e.g. "pulling something from left to right" vs "pushing something from right to left"). With 8 frames, TimeSformer likely struggles to resolve motion direction and trajectory. Classes involving subtle physical transformations (tearing, bending, squeezing, deforming) are hard without higher spatial resolution or more frames. Expect the model to confuse direction-sensitive pairs and to perform worse on classes requiring precise temporal ordering of sub-events (e.g. "plug in then pull out" vs "plug in").

### VideoMAE (Aiden)
- Pretrained from: MCG-NJU/videomae-base (ViT-B/16 self-supervised on Kinetics-400, 800 epochs)
- Fine-tuning strategy: full fine-tune, attached fresh 174-class head via HuggingFace `VideoMAEForVideoClassification.from_pretrained(..., ignore_mismatched_sizes=True)`. Forward signature `(B, T, C, H, W)` exactly matches `shared.get_dataloader` so no permute step needed (unlike VideoMamba/R(2+1)D).
- Masking ratio: N/A for fine-tuning. Original pretrain on Kinetics-400 used 90% masking with the tube-masking scheme.
- Optimizer / LR / Schedule: AdamW lr=5e-4, weight_decay=0.05, cosine annealing to 1e-6 over 15 epochs, bf16 autocast (no GradScaler needed).
- Best val epoch: 11 of 15 attempted (val top-1 0.1691). Vast.ai host paused the instance for ~24 hrs after epoch 11, training never reached epoch 15. Final canonical eval ran as a sidecar pass off `best.pt` while training was still resuming.
- What worked: HuggingFace `transformers` 5.7.0 loaded the K400 pretrain cleanly on Blackwell sm_120 with PyTorch 2.11+cu130 — no kernel-image issues like SlowFast hit on the 5090. bf16 ran end-to-end on the Pro 6000 with no GradScaler needed. Camera-pan classes were the easiest (Turning the camera left/right/up/down all >0.55, max 0.736), consistent with VideoMAE's strong bias toward dominant motion. Top-5 0.398 vs Top-1 0.169 means the model usually has the right action class in its short-list even when the head pick is wrong.
- What didn't: Training pace was 3-4x slower than my benchmark predicted (1.74 hr/epoch actual vs 0.85 hr/epoch in a clean iter() benchmark). Root cause was dataloader stalls — workers periodically blocked on slow webm decodes, so GPU sat at 0% util during those windows even though VRAM was healthy at 40.6 GB / 96 GB. More workers (16+) would probably help. Also lost ~24 hrs of wall-clock to a Vast.ai host pause that froze the process mid-epoch-12 (process state preserved, just no compute time charged or progressed).
- Failure modes: 22/174 classes at 0% accuracy. Hardest are the spilling/pouring-with-negation classes (`Spilling something behind something`, `Trying to pour something into something but missing so it spills next to it`) — these require physical-outcome reasoning the model doesn't get from RGB alone. Most-confused pair is `Tearing something just a little bit -> Tearing something into two pieces` (57 confusions); same kinematic action, the only difference is the magnitude of the tear, which is hard to pick up from 16 sampled frames. Pattern matches what V-JEPA and Qwen also struggle with: fine-grained physical-state distinctions inside a single visually-similar action template.

### VideoMamba (Kenneth)
- Pretrained from: VideoMamba-Small Kinetics-400 checkpoint released by OpenGVLab. The earlier from-scratch attempt that landed at 1.45% used the bundled `videomamba.py` from the project repo, which silently constructed a unidirectional model because the `bimamba=True` flag was being absorbed as an unused kwarg by `mamba_ssm` and had no effect on the layer construction. We integrated a real Bimamba class with `_b`-suffixed forward/backward parameters whose layout matches OpenGVLab's released checkpoint, then loaded the K400 weights into that class for fine-tuning.
- Fine-tuning strategy: full fine-tune of VideoMamba-Small from K400 init, 174-class head replacing the original 400-class head. We had budgeted 15 epochs but stopped at 2 because our Bimamba forward uses the explicit selective-scan path rather than the fused `mamba_inner_fn` kernel (we prioritized correctness over speed when matching the released checkpoint's parameter layout), making per-epoch time about 3.5 hours instead of the projected 1 hour.
- Optimizer / LR / Schedule: AdamW, cosine LR schedule, bf16 autocast, gradient scaling enabled, batch 16, 16 frames per clip.
- Best val epoch: 2 of 2 attempted. Trajectory was 50.13% top-1 at epoch 1 and 53.86% at epoch 2 (eval set produced 53.91% top-1 / 82.73% top-5 / F1 0.5290), still climbing monotonically with no saturation visible.
- What worked: identifying the silent unidirectional bug was the single biggest win in the project. The fix turned a 1.45% from-scratch run into a 53.91% K400-finetuned run in 2 epochs. Top-5 of 82.73% says the model has the right answer in its short-list more than 4 out of 5 times, which is the second-highest top-5 in the whole benchmark behind only Qwen and V-JEPA. Per-class accuracy is broad: very few classes sit at exact zero, and the model handles both camera-motion classes and several fine-grained manipulations well.
- What didn't: the selective-scan path costs us throughput. Each epoch was about 3.5x slower than the projected `mamba_inner_fn` time, which is what kept us at 2 epochs instead of 15. Conservative extrapolation puts a full 15-epoch run somewhere in the 58-62% range, above Qwen's 58.19% and approaching V-JEPA's 64.51% reference. The pre-bugfix from-scratch row in the main table is left intact as direct evidence of why correctness checks matter.
- Failure modes: not run on the new bidirectional model in the same depth as the older from-scratch one. The classes that are still hard match the pattern other RGB models hit on SSv2 (fine-grained "pretending" variants and physical-state-change classes like spilling and tearing-by-degree).

### CNN+ConvLSTM (Kenneth)
- CNN backbone: ResNet-18 (ImageNet pretrained from torchvision default weights). The ConvLSTM head sits on top of the ResNet feature maps and is trained from scratch.
- Pretrained from: ImageNet for the ResNet-18 backbone; ConvLSTM head initialized from scratch with no pretraining.
- Fine-tuning strategy: full network unlocked, 174-class linear head, 8 frames per clip, 224x224 resolution, batch 32. The ConvLSTM aggregates per-frame ResNet features over time before the classifier.
- Optimizer / LR / Schedule: standard cosine schedule over 15 epochs on a single A100 SXM4. 18.3M total params (all trainable).
- Best val epoch: epoch 14 (the curve saturates cleanly by then; epoch 15 produces the canonical eval at 27.01% top-1, 53.14% top-5, F1 0.2472).
- What worked: the model places exactly where a hybrid CNN+RNN of this size should: between TSM (6.27%) and R(2+1)D (46.36%) in the team table. Top-5 at 53.14% is roughly double the top-1, which says the right action is usually in the model's near short-list. The clean saturation by epoch 14 means we were not leaving accuracy on the table by stopping at 15 epochs. Camera-motion classes are the easiest, with "Turning the camera left while filming something" at 83.6% and the other three camera-direction classes all above 65%.
- What didn't: fine-grained object manipulation is the weak spot, just like every other RGB model on SSv2 of this scale. Classes like spilling-with-negation and pretending-to-pour sit at 0% accuracy because they require physical-outcome reasoning that 8 frames of ResNet features plus a ConvLSTM does not easily capture. The model is essentially a strong motion classifier that does not learn enough about object state.
- Failure modes: 0% accuracy on classes whose label hinges on a physical outcome rather than a motion (Spilling something behind/next to/onto something, Pretending or trying and failing to twist something, Failing to put something into something because something does not fit). Strong on classes whose label is dominated by a kinematic motion of the camera or a clear push/pull direction. Best class is "Turning the camera left while filming something" at 83.6%, and the worst class is "Trying to pour something into something but missing so it spills next to it" at 0%.

### ST-GCN (Arthur)
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
