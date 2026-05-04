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
from models.videomamba.models.videomamba_bidir import (
    videomamba_small_bidir,
    videomamba_tiny_bidir,
)

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
NUM_CLASSES = 174

BUILDERS = {
    "tiny":  videomamba_tiny_bidir,
    "small": videomamba_small_bidir,
}


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
        frames = frames.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16, enabled=use_fp16):
            logits = model(frames)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())
    if total == 0:
        raise RuntimeError("No samples evaluated")
    return correct / total, torch.cat(all_logits), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained VideoMamba bidirectional checkpoint on SSv2 val"
    )
    parser.add_argument("--model", type=str, default="small", choices=list(BUILDERS))
    parser.add_argument("--ckpt",  type=str,
                        default=os.path.join(_SCRIPT_DIR, "checkpoints", "best.pt"),
                        help="Path to a training checkpoint (best.pt or last.pt) saved by train.py")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--data-root", type=str,
                        default=os.path.join(_PROJECT_ROOT, "data", "something-something-v2"))
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--model-name", type=str, default="VideoMambaK400Finetuned",
                        help="Used for the results JSON filename")
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    use_fp16 = not args.no_fp16 and device.type == "cuda"
    print(f"Device: {device} | bf16: {use_fp16}")

    val_loader = get_dataloader(
        split="val",
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_workers=args.num_workers,
        root=args.data_root,
        annotations_dir=os.path.join(args.data_root, "annotations"),
    )
    print(f"Val samples: {len(val_loader.dataset):,} | batches: {len(val_loader)}")

    print(f"Building VideoMamba-{args.model} bidirectional...")
    model = BUILDERS[args.model](num_classes=NUM_CLASSES, num_frames=args.num_frames).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    print(f"Loading checkpoint: {args.ckpt}")
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    print(f"  loaded. missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print(f"  missing: {msg.missing_keys[:8]}")
    if msg.unexpected_keys:
        print(f"  unexpected: {msg.unexpected_keys[:8]}")

    training_time_hours = 0.0
    if isinstance(raw, dict) and "epoch" in raw:
        print(f"  checkpoint epoch: {raw['epoch'] + 1}, best_val_acc: {raw.get('best_val_acc', 'unknown')}")
    metrics_csv = os.path.join(_SCRIPT_DIR, "checkpoints", "metrics.csv")
    if os.path.isfile(metrics_csv):
        try:
            import csv as _csv
            with open(metrics_csv) as f:
                rows = list(_csv.DictReader(f))
            new_rows = [r for r in rows if float(r["val_acc"]) > 0.05]
            if new_rows:
                print(f"  metrics.csv has {len(new_rows)} valid epoch rows from this run")
        except Exception:
            pass

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
        training_time_hours=training_time_hours,
        peak_vram_gb=round(peak_vram, 2),
        total_params=total_params,
        trainable_params=total_params,
    )
    save_results(results, output_dir=RESULTS_DIR)
    print(f"\nSaved {args.model_name}_results.json + confusion matrix to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
