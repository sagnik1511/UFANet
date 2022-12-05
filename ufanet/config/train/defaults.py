from torch.cuda import is_available


# Seed Values
SEED = 42

# Data Directory Paths
DATA_DIRECTORY = "data"
TRAIN_METADATA_BASE_PATH = "train.csv"
VAL_METADATA_BASE_PATH = "val.csv"

# Dataset Configuration
REAL_WIDTH = 300
MASK_WIDTH = 212
REAL_CHANNELS = 3
MASK_CHANNELS = 1
BATCH_SIZE = 16
SHUFFLE = True
DROP_LAST = True

# Model Configuration
USE_ATTN = False
USE_FAM = True
BASE_FILTER_DIM = 32
DEPTH = 3

# Training Configuration
DEVICE = "cuda:0" if is_available() else "cpu"
NUM_EPOCHS = 2
LEARNING_RATE = 1e-4
BETAS = (0.99, 0.999)
TRACK_RESULT_COUNTER = 25
TRAIN_KILL_THRESH = 7
RESULT_FIGURE_SHAPE = (30, 9)
RESULT_WAREHOUSE_DIRECTORY = "results"
SEGM_THRESH = 0.7
