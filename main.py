# Absilute imports
import os
import glob2 as glob
import numpy as np
# Relative imports
from config import EMB_PATH, EMB_DIM, EMB_FILE
from config import DATASET_PATH, DATASET_FOLDER
from utils.load_embeddings import load_word_vectors
from utils.emodb import (parse_wav, get_mfcc_with_deltas,
                        get_indexes_for_wav_categories,
                        get_features_mean_var)
from plotting.class_stats import class_statistics


# EMBEDDINGS = os.path.join(EMB_PATH, EMB_FILE)

# word2idx, idx2word, embeddings = load_word_vectors(file=EMBEDDINGS, dim=EMB_DIM)

# Load dataset
DATASET = os.path.join(DATASET_PATH, DATASET_FOLDER)
# Check that the dataset folder exists
if not os.path.exists(DATASET):
    raise FileNotFoundError
# Get filenames
dataset_files = glob.iglob(''.join([DATASET,'*.wav']))

# Store all files
parsed_files = []
# Store features
features = []
# Parse all files and extract features
for file in dataset_files:
    # Read file using librosa and get all details
    # returning [librosa_read, speaker, phrase, emotion2idx, version]
    parsed_file = parse_wav(file)
    # Add files
    parsed_files.append(parsed_file)
    # Loaded librosa file
    librosa_loaded_file = parsed_file[0]
    # Get features
    feature = get_features_mean_var(librosa_loaded_file)
    features.append(feature) # (#samples,78)


# Create indexes
categories = get_indexes_for_wav_categories(parsed_files)
# Plot original percentages of emotion classes
class_statistics(categories, save=False)

# Get mean and variance along the second axis

