"""Qwen3.5-4B QLoRA configuration."""

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3.5-4B"
NUM_CLASSES = 174

# ─── QLoRA ────────────────────────────────────────────────────────────────────
QUANTIZATION_BITS = 4                    # 4-bit quantization via bitsandbytes
LORA_R = 16                              # LoRA rank
LORA_ALPHA = 32                          # LoRA scaling
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# ─── Data ─────────────────────────────────────────────────────────────────────
NUM_FRAMES = 8                           # frames sampled per video
FRAME_SIZE = 224
MAX_NEW_TOKENS = 10                      # max tokens for generated class label

# Prompt template -- the model sees frames + this text
PROMPT_TEMPLATE = (
    "You are watching a short video clip. The frames shown are sampled uniformly "
    "from the video. What action is being performed? Respond with ONLY the action "
    "label, nothing else."
)

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 2                           # small due to VLM memory
GRADIENT_ACCUMULATION_STEPS = 8          # effective batch = 2 * 8 = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
EPOCHS = 3                               # VLMs converge fast with LoRA
WARMUP_RATIO = 0.1

# ─── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "models/qwen/checkpoints"
RESULTS_DIR = "results"
