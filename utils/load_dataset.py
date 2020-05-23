import os
import glob2 as glob
from sklearn.model_selection import train_test_split
from utils import emodb


def load_Emodb(train_split_percentage=0.3):
    """Return X_train, y_train, X_test, y_test of EMODB dataset."""

    # Files
    DATASET_PATH = "datasets"
    DATASET_FOLDER = "emodb/wav/"
    # Load dataset
    DATASET = os.path.join(DATASET_PATH, DATASET_FOLDER)
    # Check that the dataset folder exists
    if not os.path.exists(DATASET):
        raise FileNotFoundError
    # Get filenames
    dataset_files_raw = [x for x in glob.iglob(''.join([DATASET, '*.wav']))]
    dataset_labels_raw = [emodb.parse_wav(file)[2] for file in dataset_files_raw]

    # Split to train and test set
    X_train, X_test, y_train, y_test = train_test_split(dataset_files_raw,
                                                        dataset_labels_raw,
                                                        shuffle=True,
                                                        test_size=0.3)
    return X_train, y_train, X_test, y_test