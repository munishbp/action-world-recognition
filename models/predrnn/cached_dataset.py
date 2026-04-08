"""
Dataset that loads pre-cached .npy frames from F:/ssv2_cache/ for fast training.

Each .npy file is (T, C, H, W) float32 in [0, 1]. This dataset applies
ImageNet normalization and optional augmentation, then returns tensors
ready for PredRNN.

Usage:
    from models.predrnn.cached_dataset import get_cached_dataloader

    train_loader = get_cached_dataloader(split="train", batch_size=16)
    for batch in train_loader:
        if batch is None:
            continue
        frames, labels = batch
        # frames: (B, T, C, H, W) float32, ImageNet-normalized
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.dataset import _load_labels, _load_split, IMAGENET_MEAN, IMAGENET_STD

DEFAULT_CACHE_DIR = "F:/ssv2_cache"
DEFAULT_ANNOTATIONS_DIR = os.path.join("data", "something-something-v2", "annotations")


class CachedFrameDataset(Dataset):
    """Loads pre-cached .npy frames. Fast -- no video decoding."""

    def __init__(
        self,
        split: str = "train",
        cache_dir: str = DEFAULT_CACHE_DIR,
        annotations_dir: str = DEFAULT_ANNOTATIONS_DIR,
        frame_size: int = 224,
    ):
        self.split = split
        self.cache_dir = cache_dir

        self.samples = _load_split(annotations_dir, split)
        self.label_map = _load_labels(annotations_dir)

        # Pre-filter to only videos with cached frames
        # List directory once (fast) instead of 168K os.path.exists() calls (slow on F:)
        print(f"[CachedFrameDataset] Scanning {cache_dir} for cached files...", flush=True)
        cached_ids = set()
        for f in os.listdir(cache_dir):
            if f.endswith(".npy"):
                cached_ids.add(f[:-4])  # strip .npy

        self.valid_samples = [e for e in self.samples if e["id"] in cached_ids]

        if len(self.valid_samples) < len(self.samples):
            missing = len(self.samples) - len(self.valid_samples)
            print(f"[CachedFrameDataset] {split}: {missing}/{len(self.samples)} "
                  f"missing cached frames, using {len(self.valid_samples)} samples", flush=True)

        # Transforms -- frames are already [0,1] float32, just normalize
        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(frame_size, scale=(0.6, 1.0), ratio=(0.75, 1.333)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(frame_size * 256 / 224)),
                transforms.CenterCrop(frame_size),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        entry = self.valid_samples[idx]
        video_id = entry["id"]

        try:
            frames = np.load(os.path.join(self.cache_dir, f"{video_id}.npy"))
            frames = torch.from_numpy(frames)  # (T, C, H, W) float32 [0,1]
            frames = self.transform(frames)
        except Exception:
            return None

        if self.split == "test":
            label = -1
        else:
            label_text = entry["template"].replace("[", "").replace("]", "")
            label = self.label_map.get(label_text, -1)

        return frames, label


def _skip_none_collate(batch):
    batch = [s for s in batch if s is not None]
    if len(batch) == 0:
        return None
    frames, labels = zip(*batch)
    return torch.stack(frames), torch.tensor(labels, dtype=torch.long)


def get_cached_dataloader(
    split="train",
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    cache_dir=DEFAULT_CACHE_DIR,
    annotations_dir=DEFAULT_ANNOTATIONS_DIR,
):
    dataset = CachedFrameDataset(
        split=split,
        cache_dir=cache_dir,
        annotations_dir=annotations_dir,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
        collate_fn=_skip_none_collate,
    )
