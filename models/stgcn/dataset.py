"""
Custom PyTorch Dataset for ST-GCN that loads cached skeleton keypoints.

Loads pre-extracted .npy files (from extract_keypoints.py) and adds velocity
features for direction-sensitive action recognition.

Usage:
    from models.stgcn.dataset import get_keypoint_dataloader

    train_loader = get_keypoint_dataloader(split="train", batch_size=64, num_frames=16)
    for keypoints, labels in train_loader:
        # keypoints: (B, 5, T, 33, 1) float32  -- (batch, channels, frames, joints, persons)
        # labels: (B,) int64
        ...
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Add project root to path so we can import shared
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.dataset import _load_labels, _load_split

DEFAULT_VIDEO_ROOT = os.path.join("data", "something-something-v2")
DEFAULT_ANNOTATIONS_DIR = os.path.join(DEFAULT_VIDEO_ROOT, "annotations")
DEFAULT_KEYPOINTS_DIR = os.path.join(DEFAULT_VIDEO_ROOT, "keypoints")

NUM_JOINTS = 33


def _add_velocity_features(keypoints: np.ndarray) -> np.ndarray:
    """Add per-joint velocity (dx, dy) as extra channels.

    Input:  (T, 33, 3) with channels [x, y, visibility]
    Output: (T, 33, 5) with channels [x, y, visibility, dx, dy]

    Velocity for the first frame is zero. This captures motion direction,
    which is critical for SSv2 actions like "pushing left" vs "pushing right".
    """
    T, V, C = keypoints.shape
    velocity = np.zeros((T, V, 2), dtype=np.float32)

    # dx, dy between consecutive frames
    velocity[1:] = keypoints[1:, :, :2] - keypoints[:-1, :, :2]

    # Zero out velocity where either frame has no detection (visibility ~ 0)
    for t in range(1, T):
        no_detect = (keypoints[t, :, 2] < 0.01) | (keypoints[t - 1, :, 2] < 0.01)
        velocity[t, no_detect] = 0.0

    return np.concatenate([keypoints, velocity], axis=2)  # (T, 33, 5)


class KeypointDataset(Dataset):
    """Dataset that loads cached MediaPipe keypoint .npy files for ST-GCN.

    Returns:
        keypoints: (C, T, V, M) float32 tensor
            C = 5 (x, y, visibility, dx, dy)
            T = num_frames
            V = 33 joints
            M = 1 person
        label: int64 scalar (or -1 for test split)
    """

    def __init__(
        self,
        split: str = "train",
        num_frames: int = 16,
        keypoints_dir: str = DEFAULT_KEYPOINTS_DIR,
        annotations_dir: str = DEFAULT_ANNOTATIONS_DIR,
        use_velocity: bool = True,
    ):
        self.split = split
        self.num_frames = num_frames
        self.keypoints_dir = keypoints_dir
        self.use_velocity = use_velocity

        # Load annotations (same as shared pipeline)
        self.samples = _load_split(annotations_dir, split)
        self.label_map = _load_labels(annotations_dir)

        # Pre-filter to only videos that have extracted keypoints
        self.valid_samples = []
        for entry in self.samples:
            npy_path = os.path.join(keypoints_dir, f"{entry['id']}.npy")
            if os.path.exists(npy_path):
                self.valid_samples.append(entry)

        if len(self.valid_samples) < len(self.samples):
            missing = len(self.samples) - len(self.valid_samples)
            print(f"[KeypointDataset] {split}: {missing}/{len(self.samples)} videos "
                  f"missing keypoints, using {len(self.valid_samples)} valid samples")

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        entry = self.valid_samples[idx]
        video_id = entry["id"]

        # Load cached keypoints: (T, 33, 3)
        npy_path = os.path.join(self.keypoints_dir, f"{video_id}.npy")
        keypoints = np.load(npy_path)  # (T_extracted, 33, 3)

        # Handle frame count mismatch (if extracted with different num_frames)
        T_extracted = keypoints.shape[0]
        if T_extracted != self.num_frames:
            if T_extracted > self.num_frames:
                # Subsample uniformly
                indices = np.linspace(0, T_extracted - 1, self.num_frames, dtype=int)
                keypoints = keypoints[indices]
            else:
                # Pad by repeating last frame
                pad = np.repeat(keypoints[-1:], self.num_frames - T_extracted, axis=0)
                keypoints = np.concatenate([keypoints, pad], axis=0)

        # Add velocity features: (T, 33, 3) -> (T, 33, 5)
        if self.use_velocity:
            keypoints = _add_velocity_features(keypoints)

        # Reshape for ST-GCN: (T, V, C) -> (C, T, V, M=1)
        C = keypoints.shape[2]
        keypoints = keypoints.transpose(2, 0, 1)  # (C, T, V)
        keypoints = keypoints[:, :, :, np.newaxis]  # (C, T, V, 1)

        keypoints_tensor = torch.from_numpy(keypoints).float()

        # Label
        if self.split == "test":
            label = -1
        else:
            label_text = entry["template"].replace("[", "").replace("]", "")
            label = self.label_map.get(label_text, -1)

        return keypoints_tensor, label


def get_keypoint_dataloader(
    split: str = "train",
    batch_size: int = 64,
    num_frames: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    keypoints_dir: str = DEFAULT_KEYPOINTS_DIR,
    annotations_dir: str = DEFAULT_ANNOTATIONS_DIR,
    use_velocity: bool = True,
) -> DataLoader:
    """Create a DataLoader for ST-GCN skeleton data.

    Args:
        split: One of "train", "val", "test".
        batch_size: Batch size (can be larger than RGB -- skeleton data is tiny).
        num_frames: Temporal dimension.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for GPU transfer.
        keypoints_dir: Directory containing .npy keypoint files.
        annotations_dir: Directory containing annotation JSONs.
        use_velocity: Whether to add dx/dy velocity channels (5 channels vs 3).

    Returns:
        DataLoader yielding (keypoints, labels) where:
            keypoints: (B, C, T, V, M) float32 tensor
            labels: (B,) int64 tensor
    """
    dataset = KeypointDataset(
        split=split,
        num_frames=num_frames,
        keypoints_dir=keypoints_dir,
        annotations_dir=annotations_dir,
        use_velocity=use_velocity,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )
