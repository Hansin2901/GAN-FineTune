"""
Configuration file for GAN fine-tuning
Modify these parameters according to your needs
"""

# Model paths
GENERATOR_MODEL_PATH = "generator_model.h5"
DISCRIMINATOR_MODEL_PATH = "discriminator_model.h5"

# Data paths
NEW_DATA_FOLDER = "data"  # Folder containing new MIDI files for fine-tuning
ORIGINAL_DATA_FOLDER = "archive"  # Original training data folder

# Fine-tuning parameters
FINETUNE_EPOCHS = 50  # Number of epochs for fine-tuning
BATCH_SIZE = 16  # Batch size (keep same as original training)
SAMPLE_INTERVAL = 5  # How often to print progress and generate samples
SAVE_INTERVAL = 10  # How often to save intermediate models

# Learning rates (lower than original training for stability)
GENERATOR_LR = 0.0001  # Original was 0.0002
DISCRIMINATOR_LR = 0.0001  # Original was 0.0002

# Fine-tuning strategy
USE_ORIGINAL_DATA = True  # Whether to mix original data with new data
FREEZE_EARLY_LAYERS = False  # Whether to freeze early layers of the models

# Model architecture (should match original)
SEQUENCE_LENGTH = 100
LATENT_DIMENSION = 1000

# Output settings
OUTPUT_DIR = "finetuned_models"  # Directory to save fine-tuned models
GENERATE_SAMPLES = True  # Whether to generate sample music during training
NUM_DIVERSITY_SAMPLES = 3  # Number of samples to generate for diversity evaluation

# Advanced fine-tuning options
LEARNING_RATE_DECAY = False  # Whether to apply learning rate decay
WARMUP_EPOCHS = 5  # Number of epochs with very low learning rate at start
PROGRESSIVE_UNFREEZING = False  # Whether to gradually unfreeze layers

# Data augmentation (experimental)
PITCH_SHIFT_RANGE = 0  # Range for random pitch shifting (0 = disabled)
TEMPO_VARIATION_RANGE = 0.0  # Range for tempo variation (0.0 = disabled)
