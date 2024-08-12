# config.py

import os

# --- General Configuration ---
WANDB_PROJECT = "decoder only transformer training"
WANDB_ENTITY = None # Set to your WandB username/team name, or leave None for default

# Path for saving model checkpoints
MODEL_CHECKPOINTS_DIR = "model_checkpoints"
os.makedirs(MODEL_CHECKPOINTS_DIR, exist_ok=True) # Ensure directory exists

# --- Training Hyperparameters (for main training run) ---
TRAIN_BATCH_SIZE = 16
TRAIN_CONTEXT_WINDOW = 124
TRAIN_CHANNEL_DIM = 512
TRAIN_NUM_HEADS = 8
TRAIN_NUM_LAYERS = 12
TRAIN_LEARNING_RATE = 5e-4
TRAIN_STEPS = 20000
TRAIN_VALIDATION_CHECK_EVERY = 100
TRAIN_EARLY_STOPPING_PATIENCE = 20

# --- Data Loading Configuration ---
# Set the maximum number of characters to use for training
# This limits the dataset size for faster iteration, especially for tuning.
MAX_CHARS_FOR_TRAINING = 3000000 # 3 Million characters

# --- Hyperparameter Search Grid ---
# Define lists of values to sweep over for tuning
HP_SEARCH_LRS = [1e-3, 5e-4, 1e-4]
HP_SEARCH_BATCH_SIZES = [8, 16, 32]
HP_SEARCH_CONTEXT_WINDOWS = [32, 64, 128] # Note: A smaller context window might be needed for larger batch sizes on CPU
# You can add more lists for other hyperparameters to tune
# HP_SEARCH_NUM_HEADS = [4, 8]
# HP_SEARCH_NUM_LAYERS = [4, 6]

# --- Other Configurations ---
TOKENIZER_NAME = "gpt2" # Default tokenizer