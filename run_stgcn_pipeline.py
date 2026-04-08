"""
Full ST-GCN preprocessing pipeline: download SSv2 → extract → keypoints.

Run this and walk away. It handles everything:
1. Downloads 20 x 1GB video archive parts from HuggingFace (resumable)
2. Concatenates and extracts to get individual .webm files
3. Runs MediaPipe pose keypoint extraction on all videos (resumable)

All progress is logged to stgcn_pipeline.log.

Usage:
    python run_stgcn_pipeline.py
    python run_stgcn_pipeline.py --skip-download   # if videos already exist
    python run_stgcn_pipeline.py --split val        # extract keypoints for val only
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from tqdm import tqdm


# ─── Configuration ────────────────────────────────────────────────────────────

REPO_ID = "morpheushoc/something-something-v2"
NUM_PARTS = 20
PART_TEMPLATE = "videos/20bn-something-something-v2-{:02d}"

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data", "something-something-v2")
PARTS_DIR = os.path.join(os.path.dirname(__file__), "data", "ssv2_parts")
ANNOTATIONS_DIR = os.path.join(DATA_ROOT, "annotations")
KEYPOINTS_DIR = os.path.join(DATA_ROOT, "keypoints")

LOG_FILE = os.path.join(os.path.dirname(__file__), "stgcn_pipeline.log")

NUM_LANDMARKS = 33


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ─── Step 1: Download ────────────────────────────────────────────────────────

def download_parts(log):
    """Download all 20 archive parts from HuggingFace."""
    from huggingface_hub import hf_hub_download

    os.makedirs(PARTS_DIR, exist_ok=True)

    for i in range(NUM_PARTS):
        local_name = f"20bn-something-something-v2-{i:02d}"
        local_path = os.path.join(PARTS_DIR, local_name)

        if os.path.exists(local_path) and os.path.getsize(local_path) > 900_000_000:
            log.info(f"Part {i:02d}/19: already exists ({os.path.getsize(local_path)/1e9:.2f} GB), skipping")
            continue

        log.info(f"Part {i:02d}/19: downloading...")
        start = time.time()

        try:
            cached = hf_hub_download(
                repo_id=REPO_ID,
                filename=PART_TEMPLATE.format(i),
                repo_type="dataset",
            )
            shutil.copy2(cached, local_path)
            elapsed = time.time() - start
            size_gb = os.path.getsize(local_path) / 1e9
            log.info(f"Part {i:02d}/19: done ({size_gb:.2f} GB in {elapsed:.0f}s)")
        except Exception as e:
            log.error(f"Part {i:02d}/19: FAILED - {e}")
            raise

    log.info(f"All {NUM_PARTS} parts downloaded to {PARTS_DIR}")


# ─── Step 2: Extract ─────────────────────────────────────────────────────────

def extract_videos(log):
    """Concatenate parts and extract .webm video files."""
    # Check if videos already exist
    existing = [f for f in os.listdir(DATA_ROOT) if f.endswith((".webm", ".mp4"))]
    if len(existing) > 200000:
        log.info(f"Found {len(existing)} video files already, skipping extraction")
        return

    part_files = sorted([
        os.path.join(PARTS_DIR, f)
        for f in os.listdir(PARTS_DIR)
        if f.startswith("20bn-something-something-v2-")
    ])

    if len(part_files) < NUM_PARTS:
        log.warning(f"Only {len(part_files)}/{NUM_PARTS} parts found!")

    log.info(f"Extracting {len(part_files)} parts to {DATA_ROOT}...")
    start = time.time()

    # Concatenate all parts and pipe to tar
    cat_cmd = "cat " + " ".join(f'"{p}"' for p in part_files)
    tar_cmd = f'tar zxf - -C "{DATA_ROOT}"'
    full_cmd = f"{cat_cmd} | {tar_cmd}"

    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        log.warning(f"Concatenated extraction returned code {result.returncode}: {result.stderr[:500]}")
        log.info("Attempting individual part extraction as fallback...")
        for p in part_files:
            subprocess.run(
                f'tar zxf "{p}" -C "{DATA_ROOT}"',
                shell=True, capture_output=True, text=True
            )

    elapsed = time.time() - start

    # Check for extracted files (might be in a subdirectory)
    video_files = []
    for root, dirs, files in os.walk(DATA_ROOT):
        for f in files:
            if f.endswith((".webm", ".mp4")):
                video_files.append(os.path.join(root, f))

    log.info(f"Extraction done in {elapsed/60:.1f} min, found {len(video_files)} video files")

    # If videos are in a subdirectory, move them up to DATA_ROOT
    if video_files:
        first_dir = os.path.dirname(video_files[0])
        if first_dir != DATA_ROOT and first_dir != "":
            log.info(f"Videos are in subdirectory {first_dir}, moving to {DATA_ROOT}...")
            for vf in video_files:
                dest = os.path.join(DATA_ROOT, os.path.basename(vf))
                if not os.path.exists(dest):
                    shutil.move(vf, dest)
            # Clean up empty subdirectory
            try:
                os.removedirs(first_dir)
            except OSError:
                pass

    final_count = len([f for f in os.listdir(DATA_ROOT) if f.endswith((".webm", ".mp4"))])
    log.info(f"Final video count in {DATA_ROOT}: {final_count}")

    return final_count


# ─── Step 3: Keypoint Extraction ─────────────────────────────────────────────

def extract_keypoints(split, num_frames, log):
    """Run MediaPipe pose on all videos in a split, save .npy per video."""
    import mediapipe as mp
    from decord import VideoReader, cpu

    # Add project root for shared imports
    sys.path.insert(0, os.path.dirname(__file__))
    from shared.dataset import _load_split, _sample_frame_indices

    os.makedirs(KEYPOINTS_DIR, exist_ok=True)

    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    )

    splits = ["train", "val", "test"] if split == "all" else [split]

    for current_split in splits:
        entries = _load_split(ANNOTATIONS_DIR, current_split)
        log.info(f"Keypoint extraction: {current_split} split, {len(entries)} videos, {num_frames} frames/video")

        num_processed = 0
        num_skipped = 0
        num_detected = 0
        num_failed = 0
        num_missing = 0
        failed_ids = []
        missing_ids = []

        start = time.time()

        for entry in tqdm(entries, desc=f"Keypoints ({current_split})"):
            video_id = entry["id"]
            output_path = os.path.join(KEYPOINTS_DIR, f"{video_id}.npy")

            # Resume: skip existing
            if os.path.exists(output_path):
                num_skipped += 1
                continue

            # Find video file
            video_path = None
            for ext in (".webm", ".mp4"):
                candidate = os.path.join(DATA_ROOT, f"{video_id}{ext}")
                if os.path.exists(candidate):
                    video_path = candidate
                    break

            if video_path is None:
                num_missing += 1
                missing_ids.append(video_id)
                continue

            try:
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)
                indices = _sample_frame_indices(total_frames, num_frames)
                frames = vr.get_batch(indices).asnumpy()

                keypoints = np.zeros((num_frames, NUM_LANDMARKS, 3), dtype=np.float32)
                any_detected = False

                for t in range(num_frames):
                    results = pose.process(frames[t])
                    if results.pose_landmarks is not None:
                        any_detected = True
                        for j, lm in enumerate(results.pose_landmarks.landmark):
                            keypoints[t, j, 0] = lm.x
                            keypoints[t, j, 1] = lm.y
                            keypoints[t, j, 2] = lm.visibility

                np.save(output_path, keypoints)
                num_processed += 1

                if any_detected:
                    num_detected += 1
                else:
                    num_failed += 1
                    failed_ids.append(video_id)

            except Exception as e:
                num_failed += 1
                failed_ids.append(video_id)
                if num_failed <= 10:
                    log.warning(f"Error on {video_id}: {e}")

            # Log progress every 1000 videos
            total_done = num_processed + num_skipped
            if total_done > 0 and total_done % 1000 == 0:
                elapsed = time.time() - start
                rate = num_processed / max(elapsed, 1)
                remaining = (len(entries) - total_done) / max(rate, 0.01)
                log.info(
                    f"  [{current_split}] {total_done}/{len(entries)} "
                    f"({num_processed} new, {num_skipped} skipped, {num_failed} failed) "
                    f"ETA: {remaining/3600:.1f}h"
                )

        pose.close()
        # Re-create for next split
        if current_split != splits[-1]:
            pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5,
            )

        elapsed = time.time() - start
        total_attempted = num_processed + num_failed
        detection_rate = num_detected / max(total_attempted, 1)

        log.info(
            f"Keypoints done for {current_split}: "
            f"{num_processed} processed, {num_skipped} skipped, "
            f"{num_failed} failed, {num_missing} missing video, "
            f"detection rate: {detection_rate:.1%}, "
            f"time: {elapsed/3600:.1f}h"
        )

        # Save metadata
        meta = {
            "split": current_split,
            "num_frames": num_frames,
            "num_total": len(entries),
            "num_processed": num_processed,
            "num_skipped": num_skipped,
            "num_detected": num_detected,
            "num_failed": num_failed,
            "num_missing_video": num_missing,
            "detection_rate": round(detection_rate, 4),
            "elapsed_hours": round(elapsed / 3600, 2),
            "failed_ids": failed_ids[:100],  # cap to avoid huge files
            "missing_video_ids": missing_ids[:100],
        }
        meta_path = os.path.join(KEYPOINTS_DIR, f"extraction_meta_{current_split}.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full ST-GCN preprocessing pipeline")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download+extraction (videos already exist)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip tar extraction (videos already extracted)")
    parser.add_argument("--skip-keypoints", action="store_true",
                        help="Skip keypoint extraction")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "val", "test", "all"],
                        help="Which split(s) to extract keypoints for")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="Frames per video for keypoint extraction")
    parser.add_argument("--keep-parts", action="store_true",
                        help="Keep archive parts after extraction")
    args = parser.parse_args()

    log = setup_logging()
    log.info("=" * 60)
    log.info("ST-GCN Preprocessing Pipeline Started")
    log.info("=" * 60)

    pipeline_start = time.time()

    try:
        # Step 1: Download
        if not args.skip_download:
            log.info("STEP 1: Downloading SSv2 archive parts...")
            download_parts(log)
        else:
            log.info("STEP 1: Skipped (--skip-download)")

        # Step 2: Extract
        if not args.skip_download and not args.skip_extract:
            log.info("STEP 2: Extracting videos from archives...")
            extract_videos(log)

            if not args.keep_parts and os.path.exists(PARTS_DIR):
                log.info("Cleaning up archive parts...")
                shutil.rmtree(PARTS_DIR, ignore_errors=True)
        else:
            log.info("STEP 2: Skipped")

        # Step 3: Keypoints
        if not args.skip_keypoints:
            log.info(f"STEP 3: Extracting keypoints ({args.split}, {args.num_frames} frames)...")
            extract_keypoints(args.split, args.num_frames, log)
        else:
            log.info("STEP 3: Skipped (--skip-keypoints)")

    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        raise

    total_elapsed = time.time() - pipeline_start
    log.info(f"Pipeline complete! Total time: {total_elapsed/3600:.1f} hours")
    log.info(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    main()
