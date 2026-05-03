from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from shared import evaluate_model, get_dataloader, save_results
from models.videomamba.models.videomamba import (
    videomamba_tiny,
    videomamba_small,
    videomamba_middle,
)

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
NUM_CLASSES = 174

BUILDERS = {
    "tiny":   videomamba_tiny,
    "small":  videomamba_small,
    "middle": videomamba_middle,
}


def load_ssv2_checkpoint(model: nn.Module, ckpt_path: str) -> None:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for key in ("model", "module", "state_dict", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in raw.items()}

    model_state = model.state_dict()

    # Interpolate temporal pos-embed if num_frames differs.
    pe_key_candidates = ("temporal_pos_embedding", "pos_embed", "time_embed")
    for k in pe_key_candidates:
        if k in state and k in model_state and state[k].shape != model_state[k].shape:
            src = state[k]
            tgt_shape = model_state[k].shape
            if src.dim() == 3 and tgt_shape == src.shape[:1] + (tgt_shape[1],) + src.shape[2:]:
                src_t, tgt_t = src.shape[1], tgt_shape[1]
                src = src.permute(0, 2, 1)
                src = torch.nn.functional.interpolate(src, size=tgt_t, mode="linear", align_corners=False)
                src = src.permute(0, 2, 1)
                state[k] = src
                print(f"Interpolated {k}: {tuple(raw.get(k, src).shape)} -> {tuple(src.shape)}")

    msg = model.load_state_dict(state, strict=False)
    if msg.missing_keys:
        print(f"Missing keys ({len(msg.missing_keys)}): {msg.missing_keys[:8]}{' ...' if len(msg.missing_keys) > 8 else ''}")
    if msg.unexpected_keys:
        print(f"Unexpected keys ({len(msg.unexpected_keys)}): {msg.unexpected_keys[:8]}{' ...' if len(msg.unexpected_keys) > 8 else ''}")
    if "head.weight" in msg.missing_keys or "head.bias" in msg.missing_keys:
        raise RuntimeError(
            "Classifier head missing from checkpoint. "
            "This loader is for SSv2-finetuned weights — pass an SSv2 checkpoint, not IN1K."
        )


@torch.no_grad()
def evaluate(model, loader, device, use_fp16=True):
    model.eval()
    correct = 0
    total = 0
    all_logits = []
    all_labels = []
    for batch in tqdm(loader, desc="Eval"):
        if batch is None:
            continue
        frames, labels = batch
        frames = frames.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)  # (B,C,T,H,W)
        labels = labels.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16, enabled=use_fp16):
            logits = model(frames)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())
    if total == 0:
        raise RuntimeError("No samples evaluated — every batch was None.")
    return correct / total, torch.cat(all_logits), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Eval SSv2-finetuned VideoMamba on SSv2 val")
    parser.add_argument("--model",      type=str,   default="small", choices=list(BUILDERS))
    parser.add_argument("--ckpt",       type=str,   required=True, help="Path to SSv2-finetuned .pth")
    parser.add_argument("--num-frames", type=int,   default=16, help="Match the checkpoint's f8/f16 suffix")
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--num-workers",type=int,   default=8)
    parser.add_argument("--data-root",  type=str,   default=os.path.join(_PROJECT_ROOT, "data", "something-something-v2"))
    parser.add_argument("--device",     type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-fp16",    action="store_true")
    parser.add_argument("--model-name", type=str,   default="VideoMambaSSv2Pretrained",
                        help="Used for results filename")
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    use_fp16 = not args.no_fp16 and device.type == "cuda"
    print(f"Device: {device} | bf16: {use_fp16}")

    annotations_dir = os.path.join(args.data_root, "annotations")
    val_loader = get_dataloader(
        split="val",
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_workers=args.num_workers,
        root=args.data_root,
        annotations_dir=annotations_dir,
    )
    print(f"Val samples: {len(val_loader.dataset):,} | batches: {len(val_loader)}")

    print(f"Building VideoMamba-{args.model} (num_frames={args.num_frames}, bimamba_type=v2)...")
    model = BUILDERS[args.model](
        num_classes=NUM_CLASSES,
        num_frames=args.num_frames,
        bimamba_type="v2",
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    print(f"Loading SSv2 checkpoint: {args.ckpt}")
    load_ssv2_checkpoint(model, args.ckpt)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print("\nRunning evaluation...")
    t0 = time.time()
    top1, logits, labels = evaluate(model, val_loader, device, use_fp16=use_fp16)
    elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    print(f"Top-1 (argmax): {top1:.4f}  |  time: {elapsed/60:.1f} min  |  peak VRAM: {peak_vram:.2f} GB")

    results = evaluate_model(
        logits.numpy(),
        labels.numpy(),
        model_name=args.model_name,
        training_time_hours=0.0,  # eval-only
        peak_vram_gb=round(peak_vram, 2),
        total_params=total_params,
        trainable_params=0,
    )
    save_results(results, output_dir=RESULTS_DIR)
    print(f"\nSaved {args.model_name}_results.json + confusion matrix to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
