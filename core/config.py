"""Settings and configurations"""

# Embeddings
EMB_PATH = "embeddings"
EMB_FILE = "glove.6B.50d.txt"
EMB_DIM = 50

# Dataset
DATASET = "EMODB"
FEATURE_EXTRACTOR = "MFCC"

# VOXCELEB OPTIONs
PRECOMPUTED_MELS = False

# Files
DATASET_PATH = "datasets"
DATASET_FOLDER = "emodb/wav/"

# Stored variables
VARIABLES_FOLDER = "variables/"
DETERMINISTIC = False
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


####################################################
# Training
####################################################
# Core
######
TRAINING = True
RESTORE_FROM_PREVIOUS_MODEL = False
MODEL_TO_RESTORE = "/home/dimitris/Downloads/early_stopping2.pt"
# Learning
EPOCHS = 10
VALID_FREQ = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
# Model
#######
PROJ = 256
CNN_BOOLEAN = True
# Early Stopping
################
PATIENCE = 7
DELTA = 1e-3
MODELNAME = "early_stopping.pt"
# Checkpoints & logging
#######################
CHECKPOINT_FREQ = 100
CHECKPOINT_FOLDER = "checkpoints/"
CHECKPOINT_MODELNAME = "speaker_verifier"
LOGGING = True
LOG_FILE = f'checkpoints/log_{MODELNAME[:-3]}.txt'
LOG_FILE_TEST = 'checkpoints/log_test.txt'

####################################################
# PyTorch
####################################################
# Dataloading / Speakers
NUM_WORKERS = 8
SPEAKER_N = 5
SPEAKER_M = 4
