from torch.cuda import is_available

# Seed Values
SEED = 42

# Data Directory Paths
DATA_DIRECTORY = "data/real"

# Dataset Configuration
WIDTH = 300
REAL_CHANNELS = 3
MASK_CHANNELS = 1
BATCH_SIZE = 9
SHUFFLE = True
DROP_LAST = True

# Model Configuration
USE_ATTN = False
USE_FAM = True
BASE_FILTER_DIM = 32
DEPTH = 3
CHECKPOINT_PATH = "results/best_model.pt"

# Evaluation Configuration
DEVICE = "cuda:0" if is_available() else "cpu"
GRID_SHAPE = (3, 3)
STORE_RESULTS_DIRECTORY = "evaluation"
FIG_SHAPE = (15, 15)
NUM_PATCHES = 20

