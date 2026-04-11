"""
Custom dataset for Qwen3.5-4B VLM fine-tuning on SSv2.

Reads .webm video files directly via decord (with OpenCV fallback) — no
caching needed. Converts decoded frames to PIL Images for the Qwen processor.

Usage:
    from models.qwen.dataset import SSv2VLMDataset

    dataset = SSv2VLMDataset(split="train")
    item = dataset[0]
    # item["images"]: list of PIL Images
    # item["prompt"]: instruction text
    # item["label_text"]: action class name
    # item["label_idx"]: integer class index
"""

import os
import sys

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from decord import VideoReader, cpu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.dataset import _load_labels, _load_split, _sample_frame_indices, DEFAULT_ROOT
from models.qwen import config

DEFAULT_ANNOTATIONS_DIR = os.path.join("data", "something-something-v2", "annotations")


class SSv2VLMDataset(Dataset):
    """SSv2 dataset formatted for VLM fine-tuning.

    Decodes .webm video files via decord (with OpenCV fallback) and converts
    sampled frames to PIL Images for the Qwen processor. Unreadable videos
    return None — the DataLoader's collate_fn filters them out.

    Returns dicts with:
        images: list of PIL Images (the sampled frames)
        prompt: the instruction text
        label_text: the action class name (ground truth)
        label_idx: integer class index (for evaluation)
    """

    def __init__(
        self,
        split: str = "train",
        num_frames: int = config.NUM_FRAMES,
        root: str = DEFAULT_ROOT,
        annotations_dir: str = DEFAULT_ANNOTATIONS_DIR,
        videos_dir: str | None = None,
    ):
        self.split = split
        self.num_frames = num_frames

        # Auto-detect videos subfolder (same logic as shared.dataset)
        if videos_dir is not None:
            self.videos_dir = videos_dir
        else:
            subfolder = os.path.join(root, "20bn-something-something-v2")
            self.videos_dir = subfolder if os.path.isdir(subfolder) else root

        self.samples = _load_split(annotations_dir, split)
        self.label_map = _load_labels(annotations_dir)
        self.idx_to_label = {v: k for k, v in self.label_map.items()}

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, video_id: str) -> list[Image.Image] | None:
        """Decode .webm and return list of PIL Images for sampled frames."""
        video_path = os.path.join(self.videos_dir, f"{video_id}.webm")
        if not os.path.exists(video_path):
            video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                return None

        # Attempt 1: decord (fast batch reading)
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            indices = _sample_frame_indices(total_frames, self.num_frames)
            frames_np = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) uint8
            return [Image.fromarray(frames_np[t]) for t in range(frames_np.shape[0])]
        except Exception:
            pass

        # Attempt 2: OpenCV fallback
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                return None
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return None
            indices = _sample_frame_indices(total_frames, self.num_frames)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame_bgr = cap.read()
                if not ret:
                    ret, frame_bgr = cap.read()
                if not ret:
                    cap.release()
                    return None
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            cap.release()
            return frames
        except Exception:
            return None

    def __getitem__(self, idx: int) -> dict | None:
        entry = self.samples[idx]
        video_id = entry["id"]

        frames = self._load_frames(video_id)
        if frames is None:
            return None

        if self.split == "test":
            label_text = ""
            label_idx = -1
        else:
            label_text = entry["template"].replace("[", "").replace("]", "")
            label_idx = self.label_map.get(label_text, -1)

        return {
            "images": frames,
            "prompt": config.PROMPT_TEMPLATE,
            "label_text": label_text,
            "label_idx": label_idx,
        }
