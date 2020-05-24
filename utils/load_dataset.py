import os
import glob2 as glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from utils import emodb


def load_Emodb(test_eval=[0.3,0.2], evaluation=True):
    """Return X_train, y_train, X_test, y_test of EMODB dataset."""

    # Get percentages
    test_p, eval_p = test_eval

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
    X_eval, y_eval = [], []
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

    if evaluation is False:
        return X_train_, y_train_, X_test, y_test

    # If evaluation is True split again
    sss = StratifiedShuffleSplit(n_splits=1, test_size=eval_p)
    train_idx, eval_idx = next(sss.split(X_train_,y_train_))
    # Train after both splits
    for idx in train_idx:
        X_train.append(X_train_[idx])
        y_train.append(y_train_[idx])
    # Evaluation
    for idx in eval_idx:
        X_eval.append(X_train_[idx])
        y_eval.append(y_train_[idx])

    # # Split to train and test set
    # X_train_, X_test, y_train_, y_test = train_test_split(dataset_files_raw,
    #                                                     dataset_labels_raw,
    #                                                     shuffle=True,
    #                                                     test_size=0.3)
    return X_train, y_train, X_test, y_test, X_eval, y_eval
