import os
import glob2 as glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from utils import emodb


def load_Emodb(test_val=[0.3,0.2], validation=True, oversampling=True):
    """Return X_train, y_train, X_test, y_test of EMODB dataset."""

    # Get percentages
    test_p, val_p = test_val

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
    dataset_labels_raw = [emodb.get_file_details(file)[2] for file in dataset_files_raw]

    # Initialize
    X_train_, y_train_ = [], []
    X_test, y_test = [], []
    X_train, y_train = [], []
    X_val, y_val = [], []
    # First split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_p)
    train_idx, test_idx = next(sss.split(dataset_files_raw, dataset_labels_raw))
    # Train
    for idx in train_idx:
        X_train_.append(dataset_files_raw[idx])
        y_train_.append(dataset_labels_raw[idx])
    # Test
    for idx in test_idx:
        X_test.append(dataset_files_raw[idx])
        y_test.append(dataset_labels_raw[idx])

    # Before training oversample
    # if oversampling is True:
    #     X_train_ = [[x] for x in X_train_]
    #     X_train_, y_train_ = SMOTE().fit_resample(X_train_, y_train_)

    if validation is False:
        return X_train_, y_train_, X_test, y_test

    # If valuation is True split again
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_p)
    train_idx, val_idx = next(sss.split(X_train_,y_train_))
    # Train after both splits
    for idx in train_idx:
        X_train.append(X_train_[idx])
        y_train.append(y_train_[idx])
    # validation
    for idx in val_idx:
        X_val.append(X_train_[idx])
        y_val.append(y_train_[idx])

    return X_train, y_train, X_test, y_test, X_val, y_val
