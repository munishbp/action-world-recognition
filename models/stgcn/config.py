"""ST-GCN hyperparameters and configuration."""

# ─── Data ─────────────────────────────────────────────────────────────────────
NUM_CLASSES = 174
NUM_JOINTS = 33        # MediaPipe Pose landmarks
NUM_FRAMES = 16        # temporal dimension
IN_CHANNELS = 5        # x, y, visibility, dx, dy
NUM_PERSONS = 1        # SSv2 is single-person

# ─── Model ────────────────────────────────────────────────────────────────────
TEMPORAL_KERNEL_SIZE = 9
DROPOUT = 0.5
EDGE_IMPORTANCE_WEIGHTING = True

# Channel progression for the 9 ST-GCN blocks
# (in_channels, out_channels, stride)
BLOCK_CONFIG = [
    (64,  64,  1),
    (64,  64,  1),
    (64,  64,  1),
    (64,  128, 2),
    (128, 128, 1),
    (128, 128, 1),
    (128, 256, 2),
    (256, 256, 1),
    (256, 256, 1),
]

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_DECAY_EPOCHS = [30, 40]   # multiply LR by 0.1 at these epochs
LR_DECAY_FACTOR = 0.1
WARMUP_EPOCHS = 5

# ─── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "models/stgcn/checkpoints"
RESULTS_DIR = "results"
