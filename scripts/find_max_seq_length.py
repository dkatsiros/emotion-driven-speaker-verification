# import sys
from utils.sound_processing import load_wav, get_melspectrogram
import numpy as np
from plotting.class_stats import plot_iemocap_classes_population
from utils.load_dataset import load_IEMOCAP
from utils.iemocap import get_categories_population_dictionary
# sys.path.append('../plotting')


if __name__ == "__main__":
    n_classes = 4
    # Get all sample labels
    X_train, _, X_test, _, X_val, _ = load_IEMOCAP(n_classes=n_classes)
    # Calculate and print max sequence number
    max_seq = np.max([np.shape(get_melspectrogram(load_wav(f)))[0]
                      for f in (X_train+X_test+X_val)])
    print(f"Max sequence number in IEMOCAP: {max_seq}")
