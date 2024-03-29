"""Settings and configurations"""

# Embeddings
EMB_PATH = "embeddings"
EMB_FILE = "glove.6B.50d.txt"
EMB_DIM = 50

# Dataset
DATASET = "EMODB"
FEATURE_EXTRACTOR = "MFCC"

# VOXCELEB OPTIONs
PRECOMPUTED_MELS = True

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
RESTORE_FROM_PREVIOUS_MODEL = True
# MODEL_TO_RESTORE = "checkpoints/emotional_iemocap.pt"
# TEST_FILE_PATH = 'datasets/voxceleb1/veri_test2.txt'
MODEL_TO_RESTORE = "checkpoints/voxceleb_lr=.5e-4.pt"
TEST_FILE_PATH = 'datasets/ravdess/veri_files/veri_test_exp1.2.txt'
# Learning
EPOCHS = 900
VALID_FREQ = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
# Model
#######
PROJ = 256
CNN_BOOLEAN = True
# Early Stopping
################
PATIENCE = 10
DELTA = 1e-3
MODEL = "emotional_iemocap"
MODELNAME = f"{MODEL}.pt"
# Checkpoints & logging
#######################
CHECKPOINT_FREQ = 300
CHECKPOINT_FOLDER = "checkpoints/"
CHECKPOINT_MODELNAME = f"{MODEL}"
LOGGING = True
LOG_FILE = f'checkpoints/log_{MODEL}.txt'
LOG_FILE_TEST = f'checkpoints/log_{MODEL}_test.txt'

####################################################
# PyTorch
####################################################
# Dataloading / Speakers
NUM_WORKERS = 8
SPEAKER_N = 8
SPEAKER_M = 8
SPEAKER_U = 45
