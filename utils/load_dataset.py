import os
import glob2 as glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE


def load_Emodb(test_val=[0.2, 0.2], validation=True, oversampling=True, train_only=False):
    from utils import emodb
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
    dataset_labels_raw = [emodb.get_file_details(
        file)[2] for file in dataset_files_raw]

    if train_only is True:
        return dataset_files_raw, dataset_labels_raw

    # Initialize
    X_train_, y_train_ = [], []
    X_test, y_test = [], []
    X_train, y_train = [], []
    X_val, y_val = [], []
    # First split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_p)
    train_idx, test_idx = next(
        sss.split(dataset_files_raw, dataset_labels_raw))
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
    train_idx, val_idx = next(sss.split(X_train_, y_train_))
    # Train after both splits
    for idx in train_idx:
        X_train.append(X_train_[idx])
        y_train.append(y_train_[idx])
    # validation
    for idx in val_idx:
        X_val.append(X_train_[idx])
        y_val.append(y_train_[idx])

    return X_train, y_train, X_test, y_test, X_val, y_val


def load_IEMOCAP(test_val=[0.2, 0.2], validation=True,
                 oversampling=True, n_classes=9,
                 train_only=False, SV_task=False):
    """
    Return X_train, y_train, X_test, y_test of IEMOCAP dataset.
    Warning: Labels are already integers.
    """
    from utils import iemocap

    # Minor checks
    if n_classes not in list(range(1, 10)):
        raise NameError(f"Wrong number of classes given {n_classes}.")

    # Get percentages
    test_p, val_p = test_val

    # Files
    DATASET_PATH = "datasets"
    DATASET_FOLDER = "iemocap/IEMOCAP_full_release/"
    # Load dataset
    DATASET = os.path.join(DATASET_PATH, DATASET_FOLDER)
    # Check that the dataset folder exists
    if not os.path.exists(DATASET):
        raise FileNotFoundError

    # Get all sessions
    sessions = glob.glob(''.join([DATASET, 'Session*']))
    filenames = []
    labels = []
    for session in sessions:
        evaluation_folder = os.path.join(session, 'dialog/EmoEvaluation/')
        # Get a list with all evaluation files
        evaluation_files = glob.glob(''.join([evaluation_folder, '*.txt']))
        # Each file contains the name of the wavs as well
        # as the label of each wav
        for eval_file in evaluation_files:
            filenames_, labels_ = iemocap.read_evaluation_file(
                eval_file=eval_file,
                # return speaker_id is SV_task is True
                label_value=0 if SV_task is False else 1)
            # labels += labels_
            for f, l in zip(filenames_, labels_):
                # Skip all labels > n_classes
                if l >= n_classes:
                    continue
                # Label accepted
                f_path = os.path.join(session, 'sentences', 'wav',
                                      eval_file.split(
                                          '/')[-1].replace('.txt', ''),
                                      f)
                filenames.append(''.join([f_path, '.wav']))
                labels.append(l)

    if train_only is True:
        return filenames, labels

    # Initialize
    X_train_, y_train_ = [], []
    X_test, y_test = [], []
    X_train, y_train = [], []
    X_val, y_val = [], []
    # First split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_p)
    train_idx, test_idx = next(
        sss.split(filenames, labels))
    # Train
    for idx in train_idx:
        X_train_.append(filenames[idx])
        y_train_.append(labels[idx])
    # Test
    for idx in test_idx:
        X_test.append(filenames[idx])
        y_test.append(labels[idx])

    # Before training oversample
    # if oversampling is True:
    #     X_train_ = [[x] for x in X_train_]
    #     X_train_, y_train_ = SMOTE().fit_resample(X_train_, y_train_)

    if validation is False:
        return X_train_, y_train_, X_test, y_test

    # If valuation is True split again
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_p)
    train_idx, val_idx = next(sss.split(X_train_, y_train_))
    # Train after both splits
    for idx in train_idx:
        X_train.append(X_train_[idx])
        y_train.append(y_train_[idx])
    # validation
    for idx in val_idx:
        X_val.append(X_train_[idx])
        y_val.append(y_train_[idx])

    return X_train, y_train, X_test, y_test, X_val, y_val


def load_RAVDESS(test_val=[0.2, 0.2], validation=True, oversampling=True, train_only=False, labels_only=True):
    """
    Return X_train, y_train, X_test, y_test of RAVDESS dataset.
    """
    from utils import ravdess

    # Get percentages
    test_p, val_p = test_val

    # Files
    DATASET_PATH = "datasets/"
    DATASET_FOLDER = "ravdess/"
    # Load dataset
    DATASET = os.path.join(DATASET_PATH, DATASET_FOLDER)
    # Check that the dataset folder exists
    if not os.path.exists(DATASET):
        raise FileNotFoundError

    # Get all actors
    actors = glob.glob(os.path.join(DATASET, 'Actor*'))
    filenames = []
    labels = []

    # Each filename is encoded and contains the label of each wav
    # Get all filenames into a list
    for actor in actors:
        filenames.extend(glob.glob(os.path.join(actor, '*.wav')))

    for file in filenames:
        label = ravdess.read_file_identifiers(file, labels_only=labels_only)
        labels.append(label)

    if train_only is True:
        return filenames, labels

    # Initialize
    X_train_, y_train_ = [], []
    X_test, y_test = [], []
    X_train, y_train = [], []
    X_val, y_val = [], []
    # First split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_p)
    train_idx, test_idx = next(
        sss.split(filenames, labels))
    # Train
    for idx in train_idx:
        X_train_.append(filenames[idx])
        y_train_.append(labels[idx])
    # Test
    for idx in test_idx:
        X_test.append(filenames[idx])
        y_test.append(labels[idx])

    # Before training oversample
    # if oversampling is True:
    #     X_train_ = [[x] for x in X_train_]
    #     X_train_, y_train_ = SMOTE().fit_resample(X_train_, y_train_)

    if validation is False:
        return X_train_, y_train_, X_test, y_test

    # If valuation is True split again
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_p)
    train_idx, val_idx = next(sss.split(X_train_, y_train_))
    # Train after both splits
    for idx in train_idx:
        X_train.append(X_train_[idx])
        y_train.append(y_train_[idx])
    # validation
    for idx in val_idx:
        X_val.append(X_train_[idx])
        y_val.append(y_train_[idx])

    return X_train, y_train, X_test, y_test, X_val, y_val


def load_VoxCeleb(val_ratio=0.05, validation=True):
    """
Return train_speakers, validation_speakers and test_speakers.
** `test_ratio` is fixed, unlike other datasets above. 
"""
    from random import shuffle

    # Initialize
    X_train_, y_train_ = [], []
    X_test, y_test = [], []
    X_train, y_train = [], []
    X_val, y_val = [], []

    TRAIN_FOLDER = "datasets/voxceleb1/train/wav/"
    TEST_FOLDER = "datasets/voxceleb1/test/wav/"

    # Get all speakers
    speakers = glob.glob(os.path.join(TRAIN_FOLDER, '*/'))
    shuffle(speakers)

    # Randomly select `val_ratio` validation speakers
    test_speakers = glob.glob(os.path.join(TEST_FOLDER, '*/'))

    if validation is False:
        return speakers, test_speakers, None

    train_speakers = speakers[:int(len(speakers)*(1-val_ratio))]
    val_speakers = speakers[:-int(len(speakers)*(1-val_ratio))]
    return train_speakers, test_speakers, val_speakers


def load_TIMIT(val_ratio=0.05, validation=True):
    """
Return train_speakers, validation_speakers and test_speakers.
** `test_ratio` is fixed, unlike other datasets above. 
"""
    from random import shuffle

    # Initialize
    X_train_, y_train_ = [], []
    X_test, y_test = [], []
    X_train, y_train = [], []
    X_val, y_val = [], []

    TRAIN_FOLDER = "datasets/timit/TRAIN/*/"
    TEST_FOLDER = "datasets/timit/TEST/*/"

    # Get all speakers
    speakers = glob.glob(os.path.join(TRAIN_FOLDER, '*/'))
    shuffle(speakers)

    # Randomly select `val_ratio` validation speakers
    test_speakers = glob.glob(os.path.join(TEST_FOLDER, '*/'))

    if validation is False:
        return speakers, test_speakers, None

    train_speakers = speakers[:int(len(speakers)*(1-val_ratio))]
    val_speakers = speakers[:-int(len(speakers)*(1-val_ratio))]
    return train_speakers, test_speakers, val_speakers
