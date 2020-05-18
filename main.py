# Absilute imports
import os
import glob2 as glob
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
# Relative imports
from models import knn, svm, mlp
from config import EMB_PATH, EMB_DIM, EMB_FILE
from config import DATASET_PATH, DATASET_FOLDER
from config import VARIABLES_FOLDER
from utils.sound_processing import get_features_mean_var
from utils.load_embeddings import load_word_vectors
from utils.emodb import (parse_wav,
                        get_indexes_for_wav_categories)
from plotting.class_stats import class_statistics


# EMBEDDINGS = os.path.join(EMB_PATH, EMB_FILE)

# word2idx, idx2word, embeddings = load_word_vectors(file=EMBEDDINGS, dim=EMB_DIM)

#----------------------------
# DATASET
# Load dataset
DATASET = os.path.join(DATASET_PATH, DATASET_FOLDER)
# Check that the dataset folder exists
if not os.path.exists(DATASET):
    raise FileNotFoundError
# Get filenames
dataset_files = glob.iglob(''.join([DATASET,'*.wav']))

# Try loading features
# Create paths
FEATURES_FILE = os.path.join(VARIABLES_FOLDER, 'features.pkl')
PARSED_WAV_FILE = os.path.join(VARIABLES_FOLDER, 'parsed_files.pkl')
try:
    # Load
    features = joblib.load(FEATURES_FILE)
    parsed_files = joblib.load(PARSED_WAV_FILE)
# If failed, extract them from scratch
except:
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

    # Save variables using joblib
    joblib.dump(parsed_files, FEATURES_FILE)
    joblib.dump(features, PARSED_WAV_FILE)


# Create indexes
categories = get_indexes_for_wav_categories(parsed_files)
# Plot original percentages of emotion classes
class_statistics(categories, save=True)


#----------------------------
# MODEL
# Create X:features to y:labels mapping
X = np.array(features) # (#samples,78)
y = np.array([f[3] for f in parsed_files], dtype=int) # (#samples,)

# Split to train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                     shuffle=True, test_size=0.3,random_state=0)

# Run svm classifier
svm.use(X_train, y_train, X_test, y_test, oversampling=True, pca=False)

svm.use_svm_cv(X,y,oversampling=False)