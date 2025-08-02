"""
Project configuration settings for VulBERTa-CNN with PyTorch 2.4+ compatibility
"""

import os
import torch

# 1. Core configuration
SEED = 1234
DATA_PATH = 'data'
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = VOCAB_SIZE + 2
EMBED_DIM = 768
UNK_IDX = 3
PAD_IDX = 1

# 2. Execution mode
TEST_ONLY = True  # Set to True for evaluation only
DATASET_NAME = 'devign'  # 'devign', 'd2a', or 'draper'
MODEL_TYPE = 'cnn'
CLASS_WEIGHTS = True  # Apply class weighting for imbalanced datasets
EPOCHS = 20  # Training epochs

# 3. Path configuration
LOG_FILE = 'vulberta_training.log'
MODEL_SAVE_PATH = f'./finetuning_models/VB-{MODEL_TYPE.upper()}_{DATASET_NAME}.pt'
TOKENIZER_PATH = './tokenizer'
VULBERTA_PATH = './pretraining_model/VulBERTa/'

# 4. Clang configuration (system-specific)
if os.name == 'nt':  # Windows
    CLANG_LIBRARY_PATH = 'C:/Program Files/LLVM/bin/libclang.dll'
else:  # Linux/Mac
    CLANG_LIBRARY_PATH = '/usr/lib/llvm-18/lib/libclang.so'

# 5. Device configuration (auto-detected for PyTorch 2.4+)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MULTIGPU = torch.cuda.device_count() > 1 if DEVICE.type == "cuda" else False

# 6. Advanced training parameters
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.5
USE_AMP = True  # Automatic Mixed Precision

# 7. Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 8. Log current configuration
if __name__ == "__main__":
    print(f"Current configuration:\n"
          f"Device: {DEVICE}\n"
          f"Multi-GPU: {MULTIGPU}\n"
          f"Epochs: {EPOCHS}\n"
          f"Class weights: {CLASS_WEIGHTS}\n"
          f"Dataset: {DATASET_NAME}")
