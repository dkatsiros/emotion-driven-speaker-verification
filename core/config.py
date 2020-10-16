"""Settings and configurations"""

# Embeddings
EMB_PATH = "embeddings"
EMB_FILE = "glove.6B.50d.txt"
EMB_DIM = 50

# Dataset
DATASET = "EMODB"
FEATURE_EXTRACTOR = "MFCC"

# Files
DATASET_PATH = "datasets"
DATASET_FOLDER = "emodb/wav/"

# Stored variables
VARIABLES_FOLDER = "variables/"
DETERMINISTIC = True
RECOMPUTE = True

# Sampling process
SAMPLING_RATE = 16000
WINDOW_LENGTH = round(.2 * SAMPLING_RATE)  # 200ms * samplingrate
HOP_LENGTH = WINDOW_LENGTH//2  # 50% overlap
# HOP_LENGTH = round(0.010 * 2 * SAMPLING_RATE)

# Plotting
# Folders
PLOTS_FOLDER = "plotting/plots/"
REPORTS_FOLDER = 'plotting/reports/'

# Checkpoints
CHECKPOINT_FOLDER = "checkpoints/"
CHECKPOINT_FREQ = 100
CHECKPOINT_MODELNAME = "speaker_verifier"

# Logging
LOGGING = False
LOG_FILE = 'checkpoints/log.txt'

# Early Stopping
PATIENCE = 7
DELTA = 1e-3
MODELNAME = "early_stopping.pt"
