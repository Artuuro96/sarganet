"""
Centralized configuration for the SargaNet ResNet fine-tuning pipeline.
All hyperparameters, paths, and class definitions are defined here.
"""

import os

# ─── Model Architecture ───────────────────────────────────────────────────────
MODEL_ARCH = "convnext_tiny" # "efficientnet_v2_s" or "convnext_tiny"

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_CSV = os.path.join(DATA_DIR, "labels", "labels.csv")
TEST_CSV = os.path.join(DATA_DIR, "labels", "test.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# ─── Class Definitions ────────────────────────────────────────────────────────
NUM_CLASSES = 5
CLASS_NAMES = ["nada", "bajo", "moderado", "abundante", "excesivo"]
LABEL_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ─── Image Settings ───────────────────────────────────────────────────────────
IMG_SIZE = 384       # Best resolution according to user
RESIZE_SIZE = 400    # Resize before center crop (val/test)

# ─── DataLoader Settings ──────────────────────────────────────────────────────
BATCH_SIZE = 16      # Increased to 16 since Tiny uses less VRAM
NUM_WORKERS = 4
PIN_MEMORY = True

# ─── Training: Phase 1 (Head Only) ───────────────────────────────────────────
WARMUP_EPOCHS = 5
WARMUP_LR = 1e-3

# ─── Training: Phase 2 (Full Fine-Tune) ──────────────────────────────────────
FINETUNE_EPOCHS = 50
FINETUNE_LR = 5e-5   # Lowered from 1e-4 for better convergence with pretrained weights
WEIGHT_DECAY = 0.05  # Increased for better regularization with AdamW + ConvNeXt

# ─── Training: General ───────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP_MAX_NORM = 1.0
NUM_FOLDS = 5

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42

# ─── ImageNet Normalization Stats ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
