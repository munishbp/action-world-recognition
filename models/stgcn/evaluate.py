"""
Standalone evaluation script for a trained ST-GCN model.

Loads the best checkpoint, runs inference on the validation set,
and produces results/STGCN_results.json + confusion matrix.

Usage:
    python -m models.stgcn.evaluate
    python -m models.stgcn.evaluate --checkpoint models/stgcn/checkpoints/best.pt
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.stgcn import config
from models.stgcn.model import STGCN
from models.stgcn.dataset import get_keypoint_dataloader
from shared import evaluate_model, save_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained ST-GCN")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best.pt"))
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE * 2)
    parser.add_argument("--num-frames", type=int, default=config.NUM_FRAMES)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    model = STGCN(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        temporal_kernel=config.TEMPORAL_KERNEL_SIZE,
        dropout=0.0,  # no dropout at eval
        edge_importance_weighting=config.EDGE_IMPORTANCE_WEIGHTING,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    training_time = ckpt.get("training_time_hours", None)
    peak_vram = ckpt.get("peak_vram_gb", None)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}, Val acc: {ckpt.get('val_acc', '?')}")
    print(f"  Parameters: {total_params:,}")

    # Data
    val_loader = get_keypoint_dataloader(
        split="val", batch_size=args.batch_size, num_frames=args.num_frames
    )
    print(f"Validation set: {len(val_loader.dataset)} samples")

    # Inference
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for keypoints, labels in tqdm(val_loader, desc="Evaluating"):
            logits = model(keypoints.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Evaluate with shared pipeline
    results = evaluate_model(
        all_logits,
        all_labels,
        model_name="STGCN",
        training_time_hours=training_time,
        peak_vram_gb=peak_vram,
        total_params=total_params,
        trainable_params=trainable_params,
    )
    save_results(results, output_dir=config.RESULTS_DIR)
    print(f"\nResults saved to {config.RESULTS_DIR}/STGCN_results.json")


if __name__ == "__main__":
    main()
