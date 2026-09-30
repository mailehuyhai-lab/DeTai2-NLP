"""PhoBERT transformer configuration — fixed hyperparameters."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Model
MODEL_NAME = "vinai/phobert-base-v2"

# Data
DATA_DIR = BASE_DIR / "data" / "processed"
TRAIN_CSV = DATA_DIR / "train.csv"
DEV_CSV = DATA_DIR / "dev.csv"
TEST_CSV = DATA_DIR / "test.csv"
CLASS_WEIGHTS_JSON = DATA_DIR / "class_weights.json"

# Output
MODELS_DIR = BASE_DIR / "models" / "transformer"
RESULTS_DIR = BASE_DIR / "results" / "transformer"

# Training
SEED = 42
LEARNING_RATE = 2e-5
EPOCHS = 2
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
BATCH_SIZE = 64          # CPU-optimized; larger batch = fewer iterations = faster
GRADIENT_CLIP = 1.0

# Tokenization — p90=29, p95=35-37; max_length=32 covers ~92% samples
MAX_LENGTH = 32

# Label maps
SENTIMENT_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
TOPIC_MAP = {0: "Lecturer", 1: "Training_program", 2: "Facility", 3: "Others"}
SENTIMENT_NUM_LABELS = 3
TOPIC_NUM_LABELS = 4
