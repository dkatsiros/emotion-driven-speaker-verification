# Absilute imports
import os
import glob2 as glob
# Relative imports
from config import EMB_PATH, EMB_DIM, EMB_FILE
from config import DATASET_PATH, DATASET_FOLDER
from utils.load_embeddings import load_word_vectors
from utils.emodb import get_file_details, load_emotions_mapping


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
files = []
for file in dataset_files:
    # Get filename, speaker, phrase, emotion, version
    details = get_file_details(file)
    # Skip if any mismatched files
    if details == None:
        print(f'File {file} skipped..')
        continue
    # Save file and details
    files.append([*details])

