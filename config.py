# config.py
import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model parameters
n_future = 48  # prediction sequence length
INPUT_SEQUENCE_LENGTH = 168  # input sequence length

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.00005
EPOCHS = 200

# Dataset parameters
TRAIN_SIZE = 80
VAL_SIZE = 50

# File paths
DATASET_PATH = "Datasets/DUQ_hourly.csv"  # Update this to match your dataset path