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


# Checkpoints & logging
CHECKPOINT_FOLDER = "checkpoints/"
CHECKPOINT_FREQ = 100
CHECKPOINT_MODELNAME = "speaker_verifier"
LOGGING = True
LOG_FILE = 'checkpoints/log.txt'

# Model
PROJ = 256

# Early Stopping
PATIENCE = 7
DELTA = 1e-3
MODELNAME = "early_stopping.pt"

# Training
EPOCHS = 10
LEARNING_RATE = 1e-5
VALID_FREQ = 1
CNN_BOOLEAN = True

# Dataloading / Speakers
NUM_WORKERS = 8
SPEAKER_N = 5
SPEAKER_M = 4
TOTAL_UTTERANCES_PER_SPEAKER = 40
