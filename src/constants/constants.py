import os

# Define project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Configuration file paths
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "config/config.yaml")
PARAMS_FILE_PATH = os.path.join(ROOT_DIR, "params.yaml")

# Artifacts paths
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
DATA_INGESTION_DIR = os.path.join(ARTIFACTS_DIR, "data_ingestion")
TRAINING_DIR = os.path.join(ARTIFACTS_DIR, "training")
EVALUATION_DIR = os.path.join(ARTIFACTS_DIR, "evaluation")

# Model paths
TRAINED_MODEL_PATH = os.path.join(TRAINING_DIR, "model.pth")

# Logging configuration
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
LOG_FILE_PATH = os.path.join(LOGS_DIR, "running_logs.log")

# Training parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 30
