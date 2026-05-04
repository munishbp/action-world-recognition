#!/bin/bash
# Run keypoint extraction for all splits sequentially.
# Launched as a background process - check stgcn_keypoints_*.log for progress.

PYTHON="/c/Users/munis/anaconda3/python.exe"
cd "D:/CAP 5610"

echo "=== Starting keypoint extraction pipeline ==="
echo "$(date): Starting val split..."
$PYTHON -m models.stgcn.extract_keypoints --split val --num-frames 16 2>&1 | tee stgcn_keypoints_val.log

echo "$(date): Starting train split..."
$PYTHON -m models.stgcn.extract_keypoints --split train --num-frames 16 2>&1 | tee stgcn_keypoints_train.log

echo "$(date): Starting test split..."
$PYTHON -m models.stgcn.extract_keypoints --split test --num-frames 16 2>&1 | tee stgcn_keypoints_test.log

echo "$(date): All splits complete!"
