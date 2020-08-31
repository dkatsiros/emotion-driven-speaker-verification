# import sys
from lib.sound_processing import load_wav, get_melspectrogram
import numpy as np
from core.config import WINDOW_LENGTH
# from setting import sampling rate
from plotting.class_stats import plot_iemocap_classes_population
from utils.load_dataset import load_IEMOCAP
from utils.iemocap import get_categories_population_dictionary
# sys.path.append('../plotting')


def max_sequence_length(X=None):
    """Return max sequence length for all files."""

    if X is None:
        raise AssertionError()

    # Calculate and print max sequence number
    l = [np.shape(get_melspectrogram(load_wav(f)))[0]
         for f in X]
    max_seq = np.max(l)
    # print(f"Max sequence length in dataset: {max_seq}")
    return max_seq


if __name__ == "__main__":
    n_classes = 4
    # Get all sample labels
    X_train, _, X_test, _, X_val, _ = load_IEMOCAP(n_classes=n_classes)
    # Calculate and print max sequence number
    l = [np.shape(get_melspectrogram(load_wav(f)))[0]
         for f in (X_train+X_test+X_val)]
    max_seq = np.max(l)
    print(f"Max sequence length in IEMOCAP: {max_seq}")
