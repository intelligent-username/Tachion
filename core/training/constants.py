"""
Centralized training constants for DeepAR and other models.

Import these constants instead of hardcoding values in individual files.
"""

# =============================================================================
# DeepAR Model Architecture
# =============================================================================
DEEPAR_PREDICTION_LENGTH = 24
DEEPAR_CONTEXT_LENGTH = 48
DEEPAR_NUM_LAYERS = 2
DEEPAR_HIDDEN_SIZE = 64
DEEPAR_DROPOUT_RATE = 0.1
DEEPAR_NUM_PARALLEL_SAMPLES = 100 

# =============================================================================
# DeepAR Training
# =============================================================================
DEEPAR_BATCH_SIZE = 512
DEEPAR_NUM_BATCHES_PER_EPOCH = 512 
DEEPAR_EPOCHS = 12
DEEPAR_LEARNING_RATE = 0.001
DEEPAR_WEIGHT_DECAY = 1e-8

# =============================================================================
# TFT (Temporal Fusion Transformer) Model Architecture
# =============================================================================
TFT_PREDICTION_LENGTH = 24
TFT_CONTEXT_LENGTH = 48
TFT_NUM_HEADS = 4              # Attention heads
TFT_HIDDEN_DIM = 64            # LSTM + transformer hidden size
TFT_VARIABLE_DIM = 32          # Feature embedding size
TFT_DROPOUT_RATE = 0.1

# =============================================================================
# TFT Training
# =============================================================================
TFT_BATCH_SIZE = 256
TFT_NUM_BATCHES_PER_EPOCH = 64
TFT_EPOCHS = 20
TFT_LEARNING_RATE = 0.001
TFT_WEIGHT_DECAY = 1e-8

# =============================================================================
# Shared / Runtime
# =============================================================================
DEFAULT_DEVICE = "auto"
DEFAULT_ASSET = "crypto"
DEFAULT_MODEL = "tft"          # "deepar" or "tft"
