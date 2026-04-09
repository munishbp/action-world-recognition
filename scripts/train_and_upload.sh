#!/bin/bash
# =============================================================================
# Run training, push results to GitHub, then destroy the instance.
#
# Usage:
#   bash scripts/train_and_upload.sh --batch-size 64 --num-workers 32
#
# Requires:
#   - Git configured with push access
#   - VAST_API_KEY and VAST_INSTANCE_ID env vars (for auto-destroy)
#     Get API key from: https://vast.ai/account
#     Get instance ID: vast show instances
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── 1. Run training ───────────────────────────────────────────────────────────
echo "=== Starting training ==="
bash scripts/run_training.sh "$@"

# ── 2. Push results to GitHub ─────────────────────────────────────────────────
echo ""
echo "=== Pushing results to GitHub ==="

git config user.email "vastai@training.local"
git config user.name "Vast.ai Training"

git add results/
git add models/videomamba/checkpoints/metrics.csv
git add models/videomamba/checkpoints/best.pt 2>/dev/null || echo "  best.pt too large for git, skipping"

git commit -m "Training results: $(date '+%Y-%m-%d %H:%M')" || echo "  Nothing new to commit"
git pull --rebase || echo "  Git pull --rebase failed — check for conflicts"
git push || echo "  Git push failed — check credentials"

echo "Results pushed to GitHub."

# ── 3. Destroy instance ───────────────────────────────────────────────────────
if [ -n "$VAST_API_KEY" ] && [ -n "$VAST_INSTANCE_ID" ]; then
    echo ""
    echo "=== Destroying instance $VAST_INSTANCE_ID ==="
    curl -s -X DELETE "https://console.vast.ai/api/v0/instances/$VAST_INSTANCE_ID/" \
        -H "Authorization: Bearer $VAST_API_KEY" \
        -H "Content-Type: application/json"
    echo "Instance destroy request sent."
else
    echo ""
    echo "=== VAST_API_KEY or VAST_INSTANCE_ID not set — skipping auto-destroy ==="
    echo "  Destroy manually from the Vast.ai dashboard."
fi
