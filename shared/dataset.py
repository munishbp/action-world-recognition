"""
Shared data pipeline for Something-Something V2.

Video loading: tries decord first, falls back to OpenCV on failure.
Corrupt/unreadable videos are silently skipped (warning logged).

Train transforms: RandomResizedCrop(224, scale=0.6-1.0) + ColorJitter + Normalize
Val/Test transforms: Resize(256) + CenterCrop(224) + Normalize
NO horizontal flip -- SSv2 is direction-sensitive ("pushing left" != "pushing right").

Custom transforms: pass transform= to override all defaults.

Usage:
    from shared import SomethingSomethingV2Dataset, get_dataloader

    train_loader = get_dataloader(split="train", batch_size=16, num_frames=8)
    for batch in train_loader:
        if batch is None:
            continue  # entire batch was corrupt (extremely rare)
        frames, labels = batch
        # frames: (B, T, C, H, W) float32, ImageNet-normalized
        # labels: (B,) int64
        ...
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
from torchvision import transforms

logger = logging.getLogger(__name__)


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default paths (relative to project root)
DEFAULT_ROOT = os.path.join("data", "something-something-v2")
DEFAULT_ANNOTATIONS_DIR = os.path.join(DEFAULT_ROOT, "annotations")


def _read_video_opencv(video_path: str, indices: np.ndarray) -> torch.Tensor | None:
    """Fallback video reader using OpenCV when decord fails.

    Returns:
        (T, C, H, W) float32 tensor in [0, 1], or None if unreadable.
    """
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame_bgr = cap.read()
        if not ret:
            ret, frame_bgr = cap.read()  # retry sequential read
        if ret:
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        else:
            cap.release()
            return None

    cap.release()
    frames_np = np.stack(frames)
    return torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0


def _build_default_transforms(split: str, frame_size: int) -> transforms.Compose:
    """Build split-aware transforms.

    Train: RandomResizedCrop + ColorJitter + Normalize
    Val/Test: Resize(256) + CenterCrop(frame_size) + Normalize

    NO RandomHorizontalFlip -- SSv2 is direction-sensitive.
    "Pushing left" != "Pushing right". Flipping would corrupt labels.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                frame_size,
                scale=(0.6, 1.0),
                ratio=(0.75, 1.333),
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(frame_size * 256 / 224)),
            transforms.CenterCrop(frame_size),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


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

    Loads video frames from .webm/.mp4 files, samples them uniformly in time,
    and applies split-aware transforms. Uses decord by default, falls back to
    OpenCV if decord fails (common with .webm seeking). Unreadable videos
    return None instead of crashing -- the DataLoader's collate_fn handles this.

    Default transforms (no need to set these yourself):
        Train: RandomResizedCrop(224) + ColorJitter + ImageNet Normalize
        Val/Test: Resize(256) + CenterCrop(224) + ImageNet Normalize
        NO horizontal flip (SSv2 is direction-sensitive).

    Pass transform= to override with your own transforms.

    Returns:
        (frames, label) or None if the video can't be decoded.
        frames: (T, C, H, W) float32 tensor, ImageNet-normalized
        label:  int64 scalar (-1 for test split)
    """

    def __init__(
        self,
        split: str = "train",
        num_frames: int = 8,
        frame_size: int = 224,
        root: str = DEFAULT_ROOT,
        annotations_dir: str | None = None,
        videos_dir: str | None = None,
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

        # Videos directory — auto-detect subfolder if not specified
        if videos_dir is not None:
            self.videos_dir = videos_dir
        else:
            subfolder = os.path.join(root, "20bn-something-something-v2")
            self.videos_dir = subfolder if os.path.isdir(subfolder) else root

        # Split-aware transforms (or custom override)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = _build_default_transforms(split, frame_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int] | None:
        entry = self.samples[idx]
        video_id = entry["id"]

        # Resolve video path (.webm or .mp4)
        video_path = os.path.join(self.videos_dir, f"{video_id}.webm")
        if not os.path.exists(video_path):
            video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                logger.warning(f"Video not found: {video_id}")
                return None

        frames = None

        # Attempt 1: decord (fast batch reading)
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            indices = _sample_frame_indices(total_frames, self.num_frames)
            frames_np = vr.get_batch(indices).asnumpy()
            frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
        except Exception:
            pass

        # Attempt 2: OpenCV fallback
        if frames is None:
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    if total_frames > 0:
                        indices = _sample_frame_indices(total_frames, self.num_frames)
                        frames = _read_video_opencv(video_path, indices)
                else:
                    cap.release()
            except Exception:
                pass

        if frames is None:
            logger.warning(f"All decoders failed for {video_id}")
            return None

        # Apply transforms (split-aware or custom)
        frames = self.transform(frames)

        # Label: map from text label to integer index
        if self.split == "test":
            label = -1
        else:
            label_text = entry["template"].replace("[", "").replace("]", "")
            label = self.label_map.get(label_text, -1)

        return frames, label


def _skip_none_collate(batch: list) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Filter out None samples from failed video loads, then stack."""
    batch = [s for s in batch if s is not None]
    if len(batch) == 0:
        return None
    frames, labels = zip(*batch)
    return torch.stack(frames), torch.tensor(labels, dtype=torch.long)


def get_dataloader(
    split: str = "train",
    batch_size: int = 16,
    num_frames: int = 8,
    frame_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    root: str = DEFAULT_ROOT,
    annotations_dir: str | None = None,
    videos_dir: str | None = None,
    transform: transforms.Compose | None = None,
) -> DataLoader:
    """Create a DataLoader for Something-Something V2.

    This is the main entry point. Call this, iterate over it, done.
    Augmentation, normalization, video decoding, and error handling are
    all taken care of. See module docstring for what transforms are applied.

    If a video can't be decoded (decord AND OpenCV both fail), it gets
    silently dropped from the batch. If every video in a batch fails,
    the batch comes back as None -- guard with `if batch is None: continue`.

    Args:
        split: One of "train", "val", "test".
        batch_size: Batch size.
        num_frames: Number of frames to uniformly sample per video.
            Default 8. Use 16 for transformer-based models.
        frame_size: Spatial size to resize frames to (square). Default 224.
        num_workers: Number of DataLoader worker processes. Default 4.
        pin_memory: Pin memory for faster GPU transfer. Default True.
        root: Path to video files directory.
        annotations_dir: Path to annotations directory (defaults to root/annotations).
        transform: Optional custom torchvision.transforms.Compose.
            If provided, overrides ALL default transforms (augmentation,
            resize, normalize -- everything). You're on your own.
            If not provided, split-aware defaults are used automatically.

    Returns:
        DataLoader yielding (frames, labels) or None where:
            frames: (B, T, C, H, W) float32 tensor, ImageNet-normalized
            labels: (B,) int64 tensor
    """
    dataset = SomethingSomethingV2Dataset(
        split=split,
        num_frames=num_frames,
        frame_size=frame_size,
        root=root,
        annotations_dir=annotations_dir,
        videos_dir=videos_dir,
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
        collate_fn=_skip_none_collate,
    )
