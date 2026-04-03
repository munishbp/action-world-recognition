"""
Extract pose keypoints from SSv2 videos using MediaPipe PoseLandmarker.

Uses OpenCV for video reading (more reliable with webm than decord) and
the MediaPipe Tasks API for pose estimation.

Saves per-video .npy files of shape (T, 33, 3) where channels are (x, y, visibility).
Resumable: skips videos that already have a .npy file.

Usage:
    python -m models.stgcn.extract_keypoints --split val --num-frames 16
    python -m models.stgcn.extract_keypoints --split train --num-frames 16
    python -m models.stgcn.extract_keypoints --split all --num-frames 16
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path so we can import shared
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.dataset import _load_split, _sample_frame_indices

# Default paths relative to project root
DEFAULT_VIDEO_ROOT = os.path.join("data", "something-something-v2")
DEFAULT_ANNOTATIONS_DIR = os.path.join(DEFAULT_VIDEO_ROOT, "annotations")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_VIDEO_ROOT, "keypoints")

# PoseLandmarker model
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")

NUM_LANDMARKS = 33


def ensure_model():
    """Download the PoseLandmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading PoseLandmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Downloaded ({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)")


def create_landmarker():
    """Create a MediaPipe PoseLandmarker instance."""
    import mediapipe as mp
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        num_poses=1,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def read_video_frames(video_path: str, num_frames: int) -> np.ndarray:
    """Read uniformly sampled frames from a video using OpenCV.

    Returns:
        (num_frames, H, W, 3) uint8 RGB array, or None if video can't be read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    indices = _sample_frame_indices(total_frames, num_frames)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame_bgr = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        else:
            # If seek fails, try reading sequentially from last position
            ret, frame_bgr = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            else:
                # Use a black frame as fallback
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 240
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 320
                frames.append(np.zeros((h, w, 3), dtype=np.uint8))

    cap.release()
    return np.stack(frames)


def extract_video_keypoints(
    video_path: str,
    num_frames: int,
    landmarker,
) -> tuple[np.ndarray, bool]:
    """Extract keypoints from a single video.

    Args:
        video_path: Path to .webm or .mp4 video file.
        num_frames: Number of frames to uniformly sample.
        landmarker: A MediaPipe PoseLandmarker instance.

    Returns:
        keypoints: (num_frames, 33, 3) float32 array with (x, y, visibility).
        detected: True if at least one frame had a person detected.
    """
    import mediapipe as mp

    frames = read_video_frames(video_path, num_frames)
    if frames is None:
        return np.zeros((num_frames, NUM_LANDMARKS, 3), dtype=np.float32), False

    keypoints = np.zeros((num_frames, NUM_LANDMARKS, 3), dtype=np.float32)
    any_detected = False

    for t in range(num_frames):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frames[t])
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            any_detected = True
            for j, lm in enumerate(result.pose_landmarks[0]):
                keypoints[t, j, 0] = lm.x          # normalized [0, 1]
                keypoints[t, j, 1] = lm.y          # normalized [0, 1]
                keypoints[t, j, 2] = lm.visibility  # confidence [0, 1]
        # else: row stays zeros (visibility=0 signals missing)

    return keypoints, any_detected


def get_video_ids(annotations_dir: str, split: str) -> list[str]:
    """Get all video IDs for a split (or all splits)."""
    if split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [split]

    video_ids = []
    for s in splits:
        entries = _load_split(annotations_dir, s)
        video_ids.extend(entry["id"] for entry in entries)
    return video_ids


def find_video_path(video_root: str, video_id: str) -> str | None:
    """Find the video file, trying .webm then .mp4."""
    for ext in (".webm", ".mp4"):
        path = os.path.join(video_root, f"{video_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe pose keypoints from SSv2 videos")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test", "all"],
                        help="Which split to process (default: val)")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="Frames to sample per video (default: 16)")
    parser.add_argument("--video-root", type=str, default=DEFAULT_VIDEO_ROOT,
                        help="Directory containing video files")
    parser.add_argument("--annotations-dir", type=str, default=DEFAULT_ANNOTATIONS_DIR,
                        help="Directory containing annotation JSONs")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save .npy keypoint files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ensure_model()

    landmarker = create_landmarker()

    video_ids = get_video_ids(args.annotations_dir, args.split)
    print(f"Split: {args.split} | Videos: {len(video_ids)} | Frames/video: {args.num_frames}")

    # Track stats
    num_processed = 0
    num_skipped = 0
    num_detected = 0
    num_failed_detection = 0
    num_missing_video = 0
    failed_ids = []
    missing_ids = []

    start_time = time.time()

    for video_id in tqdm(video_ids, desc=f"Extracting keypoints ({args.split})"):
        output_path = os.path.join(args.output_dir, f"{video_id}.npy")

        # Resume: skip if already extracted
        if os.path.exists(output_path):
            num_skipped += 1
            continue

        video_path = find_video_path(args.video_root, video_id)
        if video_path is None:
            num_missing_video += 1
            missing_ids.append(video_id)
            continue

        try:
            keypoints, detected = extract_video_keypoints(
                video_path, args.num_frames, landmarker
            )
            np.save(output_path, keypoints)
            num_processed += 1

            if detected:
                num_detected += 1
            else:
                num_failed_detection += 1
                failed_ids.append(video_id)

        except Exception as e:
            print(f"\nError processing {video_id}: {e}")
            failed_ids.append(video_id)
            num_failed_detection += 1

        # Log progress every 5000 videos
        total_done = num_processed + num_skipped + num_failed_detection
        if total_done > 0 and total_done % 5000 == 0:
            elapsed = time.time() - start_time
            rate = total_done / elapsed
            remaining = (len(video_ids) - total_done) / max(rate, 0.01)
            det_rate = num_detected / max(num_processed, 1)
            print(f"\n  Progress: {total_done}/{len(video_ids)}, "
                  f"rate: {rate:.1f} videos/s, "
                  f"detection: {det_rate:.1%}, "
                  f"ETA: {remaining/3600:.1f}h")

    landmarker.close()

    elapsed = time.time() - start_time
    total_attempted = num_processed + num_failed_detection

    # Save metadata
    meta = {
        "split": args.split,
        "num_frames": args.num_frames,
        "num_total_videos": len(video_ids),
        "num_processed": num_processed,
        "num_skipped_existing": num_skipped,
        "num_detected": num_detected,
        "num_failed_detection": num_failed_detection,
        "num_missing_video": num_missing_video,
        "detection_rate": num_detected / max(total_attempted, 1),
        "elapsed_seconds": round(elapsed, 1),
        "failed_ids": failed_ids[:500],
        "missing_video_ids": missing_ids[:500],
    }

    meta_path = os.path.join(args.output_dir, f"extraction_meta_{args.split}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone in {elapsed/3600:.1f} hours")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped (existing): {num_skipped}")
    print(f"  Detection rate: {meta['detection_rate']:.1%}")
    print(f"  Failed detection: {num_failed_detection}")
    print(f"  Missing video files: {num_missing_video}")
    print(f"  Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
