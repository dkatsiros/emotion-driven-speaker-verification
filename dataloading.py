import torch
import numpy as np
from torch.utils.data import Dataset
from config import DATASET, FEATURE_EXTRACTOR
from utils import emodb
from utils import sound_processing

class EmodbDataset(Dataset):
    """Custom PyTorch Dataset for preparing features from wav inputs."""


    def __init__(self, X, y, feature_extraction_method="MFCC"):
        """Create all important variables for dataset tokenization

        Arguments:
            X {list} -- [List of training samples]
            y {list} -- [List of training labels]
        """
        if feature_extraction_method == "MFCC":
            # Get file read using librosa
            X_parsed = [emodb.parse_wav(x)[0] for x in X]
            # Labels
            y = np.array([emodb.emotion2idx(lbl) for lbl in y], dtype=int)
            # Create tensor for labels
            self.y = torch.tensor(y, dtype=int)
            # X: (#samples, seq_len, #features)
            X = np.array([sound_processing.get_mfcc_with_deltas(x) for x in X_parsed])
            # Get all lengths before zero padding
            self.lengths = np.array([len(x) for x in X])
            # Zero pad all samples
            X = X = self.zero_pad_and_stack(X)
            # Create tensor for features
            self.X = torch.from_numpy(X).type('torch.FloatTensor')



    def __len__(self):
        """Returns length of EMODB dataset."""
        return len(self.X)


    def __getitem__(self, index):
        """Returns a _transformed_ item from the dataset

        Arguments:
            index {int} -- [Index of an element to be returned]

        Returns:
            (tuple)
                * sample [ndarray] -- [Features of an sample]
                * label [int] -- [Label of an sample]
                * len [int] -- [Original length of sample]
        """

        return self.X[index], self.y[index], self.lengths[index]


    def zero_pad_and_stack(self, X):
        """
        This function performs zero padding on a list of features and forms them into a numpy 3D array

        Returns:
            padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        max_length = self.lengths.max()
        feature_dim = X.shape[2]
        padded = np.zeros((len(X), max_length, feature_dim))

        # Do the actual work
        for i in range(len(X)):
            if X[i].shape[0] < max_length:
                # Needs padding
                diff = max_length - X[i].shape[0]
                # pad
                X[i] = np.vstack((X[i], np.zeros((diff, feature_dim))))
            padded[i,:,:] = X[i]
        # Return tensor
        return padded
