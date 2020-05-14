# Absilute imports
import os
import glob2 as glob
# Relative imports
from config import EMB_PATH, EMB_DIM, EMB_FILE
from config import DATASET_PATH, DATASET_FOLDER
from utils.load_embeddings import load_word_vectors
from utils.emodb import parse_wav


# EMBEDDINGS = os.path.join(EMB_PATH, EMB_FILE)

# word2idx, idx2word, embeddings = load_word_vectors(file=EMBEDDINGS, dim=EMB_DIM)

## DATASET
# Load dataset
DATASET = os.path.join(DATASET_PATH, DATASET_FOLDER)
# Check that the dataset folder exists
if not os.path.exists(DATASET):
    raise FileNotFoundError
# Get filenames
dataset_files = glob.iglob(''.join([DATASET,'*.wav']))
# Store all files
parsed_wavs = []
# Parse all files and extract features
for file in dataset_files:
    # Read file using librosa and get all details
    # returning [librosa_read, speaker, phrase, emotion2idx, version]
    parsed_file = parse_wav(file)
    parsed_wavs.append(parsed_file)

print(parsed_wavs)
