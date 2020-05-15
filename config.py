"""Settings and configurations"""

# Embeddings
EMB_PATH = "embeddings"
EMB_FILE = "glove.6B.50d.txt"
EMB_DIM = 50

# Dataset
# Files
DATASET_PATH = "datasets"
DATASET_FOLDER = "emodb/wav/"

# Sampling process
SAMPLING_RATE = 16000
WINDOW_LENGTH = round(0.025 * SAMPLING_RATE)
HOP_LENGTH = round(0.010 * SAMPLING_RATE)

# Plotting
## Folders
PLOTS_FOLDER = "plotting/plots/"