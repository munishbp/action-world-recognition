"""
Training script for Qwen3.5-4B with QLoRA on SSv2.

Loads the model in 4-bit quantization, applies LoRA adapters,
and fine-tunes on video action classification.

Usage:
    python -m models.qwen.train
    python -m models.qwen.train --epochs 3 --batch-size 2
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.qwen import config
from models.qwen.dataset import SSv2VLMDataset


def build_messages(images, prompt):
    """Build chat messages with video frames for Qwen's processor."""
    # Each frame as an image in the conversation
    image_content = [{"type": "image", "image": img} for img in images]
    image_content.append({"type": "text", "text": prompt})

    return [{"role": "user", "content": image_content}]


def collate_fn(batch):
    """Filter None and return list of dicts (no stacking -- VLM handles variable sizes)."""
    return [item for item in batch if item is not None]


def train_one_epoch(model, processor, loader, optimizer, device, grad_accum_steps):
    model.train()
    total_loss = 0.0
    total_steps = 0
    optimizer.zero_grad()

    for step, batch_items in enumerate(tqdm(loader, desc="  Train", leave=False)):
        if len(batch_items) == 0:
            continue

        batch_loss = 0.0
        for item in batch_items:
            messages = build_messages(item["images"], item["prompt"])
            label_text = item["label_text"]

            # Build input with the processor
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text + label_text],
                images=item["images"],
                return_tensors="pt",
                padding=True,
            ).to(device)

            # Create labels: mask everything except the answer tokens
            labels = inputs["input_ids"].clone()
            # Find where the answer starts (after the prompt)
            prompt_inputs = processor(
                text=[text],
                images=item["images"],
                return_tensors="pt",
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels[:, :prompt_len] = -100  # mask prompt tokens

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            batch_loss += outputs.loss.item()

        total_loss += batch_loss / len(batch_items)
        total_steps += 1

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    # Final gradient step if needed
    if total_steps % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate(model, processor, loader, label_map, device):
    """Run evaluation: generate predictions and compare to ground truth."""
    model.eval()
    idx_to_label = {v: k for k, v in label_map.items()}

    all_preds = []
    all_labels = []

    for batch_items in tqdm(loader, desc="  Eval", leave=False):
        if len(batch_items) == 0:
            continue

        for item in batch_items:
            messages = build_messages(item["images"], item["prompt"])
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=item["images"],
                return_tensors="pt",
            ).to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=False,
            )

            # Decode only the generated tokens
            generated = processor.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )[0].strip()

            # Match generated text to closest class label
            pred_idx = match_label(generated, label_map)
            all_preds.append(pred_idx)
            all_labels.append(item["label_idx"])

    return np.array(all_preds), np.array(all_labels)


def match_label(generated_text: str, label_map: dict) -> int:
    """Match generated text to the closest class label.

    Tries exact match first, then case-insensitive substring match.
    Falls back to -1 if no match found.
    """
    generated_lower = generated_text.lower().strip()

    # Exact match
    if generated_text in label_map:
        return label_map[generated_text]

    # Case-insensitive exact match
    for label, idx in label_map.items():
        if label.lower() == generated_lower:
            return idx

    # Substring match (generated text contains the label or vice versa)
    best_match = -1
    best_len = 0
    for label, idx in label_map.items():
        label_lower = label.lower()
        if label_lower in generated_lower or generated_lower in label_lower:
            if len(label) > best_len:
                best_len = len(label)
                best_match = idx

    return best_match


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3.5-4B QLoRA on SSv2")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--num-frames", type=int, default=config.NUM_FRAMES)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load processor
    print(f"Loading {config.MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(config.MODEL_ID, trust_remote_code=True)

    # Load model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        config.MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for QLoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} total, {trainable:,} trainable ({100*trainable/total:.2f}%)")

    # Data
    print("Loading data...")
    train_dataset = SSv2VLMDataset(split="train", num_frames=args.num_frames)
    val_dataset = SSv2VLMDataset(split="val", num_frames=args.num_frames)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn,
    )
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Checkpoint dir + CSV log
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    metrics_path = os.path.join(config.CHECKPOINT_DIR, "metrics.csv")
    metrics_fields = ["epoch", "train_loss", "val_acc", "best_val_acc"]
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=metrics_fields).writeheader()

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    training_start = time.time()
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(
            model, processor, train_loader, optimizer, device,
            config.GRADIENT_ACCUMULATION_STEPS,
        )
        print(f"  Train loss: {train_loss:.4f}")

        # Evaluate
        print("  Running evaluation...")
        preds, labels = evaluate(model, processor, val_loader, train_dataset.label_map, device)
        val_acc = (preds == labels).mean()
        print(f"  Val acc: {val_acc:.4f}")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"  New best! {best_val_acc:.4f}")
            model.save_pretrained(os.path.join(config.CHECKPOINT_DIR, "best"))
            processor.save_pretrained(os.path.join(config.CHECKPOINT_DIR, "best"))

        # CSV log
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=metrics_fields).writerow({
                "epoch": epoch + 1,
                "train_loss": f"{train_loss:.4f}",
                "val_acc": f"{val_acc:.4f}",
                "best_val_acc": f"{best_val_acc:.4f}",
            })

    training_time = time.time() - training_start
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    print(f"\nTraining complete!")
    print(f"  Time: {training_time / 3600:.1f} hours")
    print(f"  Best val acc: {best_val_acc:.4f}")
    print(f"  Peak VRAM: {peak_vram:.2f} GB")

    # Final eval with shared pipeline
    print("\nRunning final evaluation with shared pipeline...")
    from shared import evaluate_model, save_results

    # Reload best adapter
    from peft import PeftModel
    base_model = AutoModelForImageTextToText.from_pretrained(
        config.MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, os.path.join(config.CHECKPOINT_DIR, "best"))

    preds, labels = evaluate(model, processor, val_loader, train_dataset.label_map, device)

    results = evaluate_model(
        preds,
        labels,
        model_name="Qwen3.5-4B",
        training_time_hours=round(training_time / 3600, 2),
        peak_vram_gb=round(peak_vram, 2),
        total_params=total,
        trainable_params=trainable,
    )
    save_results(results, output_dir=config.RESULTS_DIR)
    print("Results saved to results/Qwen3.5-4B_results.json")


if __name__ == "__main__":
    main()
