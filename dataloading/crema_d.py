import os
import torch
from imblearn.over_sampling import RandomOverSampler
from lib import sound_processing
from utils import iemocap
from core.config import DATASET, FEATURE_EXTRACTOR
from torch.utils.data import Dataset
import numpy as np
import glob2 as glob
from random import shuffle
from core import config


class CremaDDataset(Dataset):
    """Custom PyTorch Dataset for preparing features from wav inputs."""

    def __init__(self, X, y, feature_extraction_method="MFCC",
                 oversampling=False, fixed_length=False,
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
            # (features,seq_len) -> (seq_len,features)
            # X = [np.swapaxes(x, 0, 1) for x in X_parsed]
            sound_processing.preview_melspectrogram(X[1])

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
        # y = np.array([iemocap.emotion2idx(lbl) for lbl in y], dtype=int)

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
            if self.max_seq_len is None:
                max_length = 342  # 320  # 998  # self.lengths.max()
            else:
                max_length = self.max_seq_len

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
        raise NotImplementedError(
            "Zero padding works only with mel spectrogram for now")


class CremaDPrecomputed(Dataset):
    """Fast implementation of PyTorch's abstruct dataset for CremaD SV task"""

    def __init__(self, X, training=False, validation=False, test=False,
                 max_seq_len=None, fixed_length=True, fe_method="MEL_SPECTROGRAM",
                 dataset_path='datasets/crema-d/AudioWAV',
                 *args, **kwargs):
        # Only one can be True
        assert (training + validation + test) == 1
        # Number of speakers per batch
        self.N = config.SPEAKER_N
        # Number of utterances per speaker per batch
        self.M = config.SPEAKER_M
        # Number of utterances per speaker in all batches
        self.U = config.SPEAKER_U
        self.utterance_indexes = list(range(self.U))

        self.max_seq_len = max_seq_len
        self.fixed_length = fixed_length
        self.fe_method = fe_method
        self.dataset_path = dataset_path

        if training is True:
            self.speakers = X
            self.S = len(self.speakers)
            shuffle(self.speakers)
            shuffle(self.utterance_indexes)
            return

        if validation is True:
            self.speakers = X
            self.S = len(self.speakers)
            shuffle(self.speakers)
            shuffle(self.utterance_indexes)
            return

        if test is True:
            self.speakers = X  # Test
            self.S = len(self.speakers)
            shuffle(self.speakers)
            shuffle(self.utterance_indexes)
            return

    def __len__(self):
        # Custom length = total_#speakers * (U/M)
        # total number of speakers != N
        return len(self.speakers) * (self.U // self.M)

    def __getitem__(self, idx):
        # # Shuffling at the beggining of each epoch
        if idx == 0:
            shuffle(self.speakers)
            shuffle(self.utterance_indexes)

        # Do not shuffle in Dataloader.
        # Shuffling is implemented in Dataset (__init__)

        # So, in each epoch spk_i is accessed U/M times
        # ensuring that all U samples of i'th speaker are used
        curr_speaker_idx = idx % self.S
        # k represents the round or (U)div(S)
        # where S in the total number of speakers
        k = idx // self.S

        # Get the speaker and all his wav's
        speaker = self.speakers[curr_speaker_idx]

        # assert that glob.glob() reads each time the files
        # in the same order. that ensures that each file is being
        # read only once in each epoch
        wav_files = glob.glob(os.path.join(self.dataset_path,
                                           str(speaker + 1000)) + "*.npy")

        # wav_files = glob.glob(speaker+'/*/*.npy')
        # shuffle wav files
        wav_files = [wav_files[i] for i in self.utterance_indexes]

        # We want to get only a specific subset of
        # the wav_files. This round `k` get [k*M:(k+1)*M]
        wav_files = wav_files[k * self.M:(k+1) * self.M]

        # Just read the input np arrays. 0.036sec to 0.0093 per wav faster
        # ~75% relative increase in bottleneck
        features = [np.load(wav) for wav in wav_files]
        features = self.zero_pad_and_stack(features)
        return torch.Tensor(features)

    def zero_pad_and_stack(self, X):
        """
        This function performs zero padding on a list of features and forms them into a numpy 3D array

        Returns:
            padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """

        if self.fe_method == "MEL_SPECTROGRAM":
            max_length = self.max_seq_len

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
        raise NotImplementedError(
            "Zero padding works only with mel spectrogram for now")


class CremaD_Evaluation_PreComputedMelSpectr(Dataset):
    def __init__(self, test_file_path="datasets/crema-d/emotional_speaker_verification_exported/veri.txt",
                 max_seq_len=46, fixed_length=True, fe_method="MEL_SPECTROGRAM",
                 *args, **kwargs):
        self.fe_method = fe_method
        self.fixed_length = fixed_length
        self.max_seq_len = max_seq_len
        # Read the data from file
        with open(test_file_path, mode="r") as file:
            data_raw = file.readlines()
        data = [l.replace('\n', '').split() for l in data_raw]
        # save length
        self.n_evaluations = len(data)
        # Get label,u1,u2
        labels, files1, files2 = zip(*data)
        # create absolute paths
        path_to_test = 'datasets/crema-d/AudioWAV'
        self.files1 = [os.path.join(path_to_test, f.strip()) for f in files1]
        self.files2 = [os.path.join(path_to_test, f.strip()) for f in files2]
        # convert to npy paths
        self.files1 = list(map(lambda x: x + '_mel.npy', self.files1))
        self.files2 = list(map(lambda x: x + '_mel.npy', self.files2))
        # make sure that all files exist and that none path is broken
        assert(all([os.path.exists(file) for file in self.files1+self.files2]))
        # typecasting str to integers
        self.labels = np.array(list(map(int, labels)), dtype=np.int8)
        assert all([l in [0, 1] for l in self.labels])

    def __len__(self):
        return self.n_evaluations

    def __getitem__(self, idx):

        label, u1, u2 = self.labels[idx], self.files1[idx], self.files2[idx]
        # load both utterances for npy precomputed format and zero pad
        features = self.zero_pad_and_stack([np.load(u) for u in [u1, u2]])
        # return 2 tensors
        # input_features(u1,u2) and label
        return torch.tensor(features), torch.tensor(label)

    def zero_pad_and_stack(self, X):
        """
        This function performs zero padding on a list of features and forms them into a numpy 3D array

        Returns:
            padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """

        if self.fe_method == "MEL_SPECTROGRAM":
            max_length = self.max_seq_len

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
        raise NotImplementedError(
            "Zero padding works only with mel spectrogram for now")
