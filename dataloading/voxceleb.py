import torch
from imblearn.over_sampling import RandomOverSampler
from lib import sound_processing
from utils import iemocap
from core.config import DATASET, FEATURE_EXTRACTOR
from torch.utils.data import Dataset
import numpy as np


class VoxcelebDataset(Dataset):
    """Custom PyTorch Dataset for preparing features from wav inputs."""

    def __init__(self, X, y,
                 feature_extraction_method="MFCC",
                 oversampling=False,
                 fixed_length=True,
                 max_seq_len=None):
        """Create all important variables for dataset tokenization

        Arguments:
            X {list} -- [List of training samples]
            y {list} -- [List of training labels]
            feature_extraction_method {string} -- [What method extracts the features]
            oversampling {bool} -- [Resampling technique to be applied]
        """
        self.feature_extraction_method = feature_extraction_method
        self.fixed_length = fixed_length
        self.max_seq_len = max_seq_len

        # Make sure there is max_seq_len
        if max_seq_len is None:
            raise AssertionError()

        if oversampling is True:
            ros = RandomOverSampler()
            # Expand last dimension
            X, y = ros.fit_resample(np.reshape(X, (len(X), -1)), y)
            # Reshape again for use
            X = np.squeeze(X)

        # Depending on the extraction method get X
        if feature_extraction_method == "MEL_SPECTROGRAM":
            # Get file read using librosa
            X_parsed = [iemocap.parse_wav(x) for x in X]
            # Get spectrogram
            X = [sound_processing.get_melspectrogram(x) for x in X_parsed]

        elif feature_extraction_method == "MFCC":
            # Get file read using librosa
            X_parsed = [iemocap.parse_wav(x) for x in X]
            # Get features
            X = [sound_processing.get_mfcc_with_deltas(x) for x in X_parsed]
            # X: (#samples, seq_len, #features)
            X = [np.swapaxes(x, 0, 1) for x in X]
        else:
            raise NameError(
                f'Not known method of feature extraction: {feature_extraction_method}')
        # Labels

        # Create tensor for labels
        self.y = torch.tensor(y, dtype=int)
        # Get all lengths before zero padding
        lengths = np.array([len(x) for x in X])
        self.lengths = torch.tensor(lengths)

        # Zero pad all samples
        X = self.zero_pad_and_stack(X)
        # Create tensor for features
        self.X = torch.from_numpy(X).type('torch.FloatTensor')
        print(np.shape(self.X))

    def __len__(self):
        """Returns length of current Dataset samples ~ X."""
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
        # print(self.X[index].size(), self.y[index].size(), self.lengths[index].size())
        return self.X[index], self.y[index], self.lengths[index]

    def zero_pad_and_stack(self, X,):
        """
        This function performs zero padding on a list of features and forms them into a numpy 3D array

        Returns:
            padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        if self.feature_extraction_method == "MFCC":
            max_length = self.lengths.max()

            feature_dim = X[0].shape[-1]
            padded = np.zeros((len(X), max_length, feature_dim))

            # Do the actual work
            for i in range(len(X)):
                if X[i].shape[0] < max_length:
                    # Needs padding
                    diff = max_length - X[i].shape[0]
                    # pad
                    X[i] = np.vstack((X[i], np.zeros((diff, feature_dim))))
                padded[i, :, :] = X[i]
            return padded

        elif self.feature_extraction_method == "MEL_SPECTROGRAM":
            max_length = self.max_seq_len  # 320  # 998  # self.lengths.max()

            feature_dim = X[0].shape[-1]
            padded = np.zeros((len(X), max_length, feature_dim))

            # Do the actual work
            for i in range(len(X)):
                if X[i].shape[0] < max_length:
                    # Needs padding
                    diff = max_length - X[i].shape[0]
                    # pad
                    X[i] = np.vstack((X[i], np.zeros((diff, feature_dim))))
                else:
                    if self.fixed_length is True:
                        # Set a fixed length => information loss
                        X[i] = np.take(X[i], list(
                            range(0, max_length)), axis=0)
                # Add to padded
                padded[i, :, :] = X[i]
            return padded

        else:
            raise AssertionError()
