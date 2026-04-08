"""PredRNN hyperparameters and configuration."""

# ─── Data ─────────────────────────────────────────────────────────────────────
NUM_CLASSES = 174
NUM_FRAMES = 8
FRAME_SIZE = 224

# ─── Model ────────────────────────────────────────────────────────────────────
ENCODER_CHANNELS = [32, 64, 64]          # CNN encoder: 224 -> 112 -> 56 -> 28
STLSTM_CHANNELS = [64, 64, 128, 128]    # 4 ST-LSTM layers
MEMORY_CHANNELS = 128                     # Spatial memory M width (uniform across layers)
STLSTM_KERNEL = 5                        # Conv kernel in ST-LSTM gates
DROPOUT = 0.3

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
GRAD_CLIP = 1.0

# ─── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "models/predrnn/checkpoints"
RESULTS_DIR = "results"
