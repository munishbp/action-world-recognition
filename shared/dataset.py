"""
Shared data pipeline for Something-Something V2.

Usage:
    from shared import SomethingSomethingV2Dataset, get_dataloader

    train_loader = get_dataloader(split="train", batch_size=16, num_frames=8)
    for frames, labels in train_loader:
        # frames: (B, T, C, H, W) float32, ImageNet-normalized
        # labels: (B,) int64
        ...
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
from torchvision import transforms


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default paths (relative to project root)
DEFAULT_ROOT = os.path.join("data", "something-something-v2")
DEFAULT_ANNOTATIONS_DIR = os.path.join(DEFAULT_ROOT, "annotations")


def _load_labels(annotations_dir: str) -> dict[str, int]:
    """Load label name -> integer index mapping from labels JSON."""
    labels_path = os.path.join(annotations_dir, "something-something-v2-labels.json")
    with open(labels_path, "r") as f:
        label_map = json.load(f)
    # labels.json maps label string -> string index, e.g. {"Pushing something": "0", ...}
    return {k: int(v) for k, v in label_map.items()}


def _load_split(annotations_dir: str, split: str) -> list[dict]:
    """Load annotation entries for a given split."""
    split_files = {
        "train": "something-something-v2-train.json",
        "val": "something-something-v2-validation.json",
        "test": "something-something-v2-test.json",
    }
    if split not in split_files:
        raise ValueError(f"split must be one of {list(split_files.keys())}, got '{split}'")

    path = os.path.join(annotations_dir, split_files[split])
    with open(path, "r") as f:
        return json.load(f)


def _sample_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    """Uniformly sample `num_frames` indices from a video of `total_frames`.

    Divides the video into `num_frames` equal segments and picks the
    center frame of each segment. Handles edge cases where total_frames
    < num_frames by repeating/padding.
    """
    if total_frames <= 0:
        return np.zeros(num_frames, dtype=np.int64)

    if total_frames < num_frames:
        # Repeat frames to fill the required count
        indices = np.arange(total_frames)
        pad = np.random.choice(total_frames, num_frames - total_frames, replace=True)
        indices = np.sort(np.concatenate([indices, pad]))
        return indices

    # Divide into equal segments, take center of each
    seg_size = total_frames / num_frames
    indices = np.array([
        int(seg_size * i + seg_size / 2) for i in range(num_frames)
    ])
    return np.clip(indices, 0, total_frames - 1)


class SomethingSomethingV2Dataset(Dataset):
    """PyTorch Dataset for Something-Something V2.

    Returns:
        frames: (T, C, H, W) float32 tensor, ImageNet-normalized
        label:  int64 scalar tensor (or -1 for test split)
    """

    def __init__(
        self,
        split: str = "train",
        num_frames: int = 8,
        frame_size: int = 224,
        root: str = DEFAULT_ROOT,
        annotations_dir: str | None = None,
        transform: transforms.Compose | None = None,
    ):
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.root = root
        self.annotations_dir = annotations_dir or os.path.join(root, "annotations")

        # Load annotations
        self.samples = _load_split(self.annotations_dir, split)
        self.label_map = _load_labels(self.annotations_dir)

        # Videos directory
        self.videos_dir = root

        # Transform: resize + normalize
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((frame_size, frame_size)),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        entry = self.samples[idx]
        video_id = entry["id"]

        # Try .webm first, then .mp4 (some setups convert to mp4)
        video_path = os.path.join(self.videos_dir, f"{video_id}.webm")
        if not os.path.exists(video_path):
            video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")

        # Decode video frames with decord
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        indices = _sample_frame_indices(total_frames, self.num_frames)

        # (num_frames, H, W, C) uint8 numpy array
        frames = vr.get_batch(indices).asnumpy()

        # Convert to (T, C, H, W) float32 tensor in [0, 1]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        # Apply per-frame transforms (resize + normalize)
        frames = self.transform(frames)

        # Label: map from text label to integer index
        if self.split == "test":
            label = -1
        else:
            label_text = entry["template"].replace("[", "").replace("]", "")
            label = self.label_map.get(label_text, -1)

        return frames, label


def get_dataloader(
    split: str = "train",
    batch_size: int = 16,
    num_frames: int = 8,
    frame_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    root: str = DEFAULT_ROOT,
    annotations_dir: str | None = None,
    transform: transforms.Compose | None = None,
) -> DataLoader:
    """Create a DataLoader for Something-Something V2.

    Args:
        split: One of "train", "val", "test".
        batch_size: Batch size.
        num_frames: Number of frames to uniformly sample per video.
        frame_size: Spatial size to resize frames to (square).
        num_workers: Number of DataLoader worker processes.
        pin_memory: Pin memory for faster GPU transfer.
        root: Path to video files directory.
        annotations_dir: Path to annotations directory (defaults to root/annotations).
        transform: Optional custom transform (overrides default resize+normalize).

    Returns:
        DataLoader yielding (frames, labels) where:
            frames: (B, T, C, H, W) float32 tensor
            labels: (B,) int64 tensor
    """
    dataset = SomethingSomethingV2Dataset(
        split=split,
        num_frames=num_frames,
        frame_size=frame_size,
        root=root,
        annotations_dir=annotations_dir,
        transform=transform,
    )

    shuffle = (split == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )
