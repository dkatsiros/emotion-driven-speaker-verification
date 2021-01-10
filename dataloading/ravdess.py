import os
import numpy as np
import glob2 as glob
from random import shuffle
from copy import deepcopy

import torch
from torch.utils.data import Dataset
from imblearn.over_sampling import RandomOverSampler

from utils import iemocap
from lib import sound_processing
from lib.sound_processing import load_wav, get_melspectrogram
from lib.sound_processing import mfccs_and_spec
from core.config import DATASET, FEATURE_EXTRACTOR
from core import config


class RAVDESS_Evaluation_PreComputedMelSpectr(Dataset):
    def __init__(self, test_file_path="datasets/ravdess/veri_files/veri_test_exp1.1.txt",
                 path_to_speakers='',
                 max_seq_len=None, fixed_length=True, fe_method="MEL_SPECTROGRAM",
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
        path_to_test = 'datasets/ravdess/'
        self.files1 = [os.path.join(path_to_test, f.strip()) for f in files1]
        self.files2 = [os.path.join(path_to_test, f.strip()) for f in files2]
        # convert to npy paths
        self.files1 = list(map(lambda x: x[:-4]+'_mel.npy', self.files1))
        self.files2 = list(map(lambda x: x[:-4]+'_mel.npy', self.files2))
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
