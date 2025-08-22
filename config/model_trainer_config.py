import torch

# Paths
TRAIN_PATH = "artifacts/processed/train"
VAL_PATH = "artifacts/processed/val"
TEST_PATH = "artifacts/processed/test"

# Image Processing
IMG_SIZE = (224, 224)

# Training
BATCH_SIZE = 32
EPOCHS = 1
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
NUM_CLASSES = 6

# Model Saving path
WARMUP_MODEL_PATH = "artifacts/models/warmup_models"
FINETUNE_MODEL_PATH = "artifacts/models/finetune_models"
OVERALL_BEST_MODEL_PATH = "artifacts/models/overallbest_model"

TENSORBOARD_LOG_PATH = "runs"