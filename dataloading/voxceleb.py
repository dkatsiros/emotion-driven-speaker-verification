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


class Voxceleb1(Dataset):
    """Fast implementation of PyTorch's abstruct dataset for Voxceleb"""

    def __init__(self, X, training=False, validation=False, test=False,
                 fe_method=None, max_seq_len=None, fixed_length=True,
                 * args, **kwargs):
        # Only one can be True
        assert (training + validation + test) == 1
        self.fe_method = fe_method
        self.max_seq_len = max_seq_len
        self.fixed_length = fixed_length

        if training is True:
            self.path = 'datasets/voxceleb1/train/wav'
            self.speakers = X
            self.utterance_number = config.SPEAKER_M
            shuffle(self.speakers)
            return

        if validation is True:
            self.path = 'datasets/voxceleb1/validation/wav'
            self.speakers = X
            self.utterance_number = config.SPEAKER_M
            shuffle(self.speakers)
            return

        if test is True:
            self.path = 'datasets/voxceleb1/test/wav'
            self.speakers = X
            self.utterance_number = config.SPEAKER_M  # Test
            shuffle(self.speakers)
            return

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):

        # Get the speaker and all his wav's
        speaker = self.speakers[idx]
        wav_files = glob.glob(speaker+'/*/*.wav')

        # At each call, different samples of the same speaker
        # are given to the dataloader. Every epoch the model
        # pottentially learns different samples for the same speaker.
        shuffle(wav_files)
        wav_files = wav_files[0:self.utterance_number]

        if self.fe_method is None:
            raise NotImplementedError("fe_method is missing!")
        # feature extraction method
        if self.fe_method == "MFCC":
            mel_dbs = []
            for f in wav_files:
                _, mel_db, _ = mfccs_and_spec(f, wav_process=True)
                mel_dbs.append(mel_db)
        if self.fe_method == "MEL_SPECTROGRAM":
            features = list(map(lambda x: get_melspectrogram(load_wav(x)),
                                wav_files))
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


class VoxCeleb(Dataset):
    """Fast implementation of PyTorch's abstruct dataset for Voxceleb"""

    def __init__(self, training=False, validation=False, test=False, *args, **kwargs):
        # Only one can be True
        assert (training + validation + test) == 1

        if training is True:
            self.path = 'datasets/voxceleb1/train/wav'
            self.speakers = glob.glob(''.join([self.path, '/*']))
            # Compute min length of utternaces per speaker
            # self.utterance_number = min([
            #     len(glob.glob(sp+'/*/*.wav')) for sp in self.speakers])
            # config.SPEAKER_M = self.utterance_number
            self.utterance_number = config.SPEAKER_M
            shuffle(self.speakers)
            return

        if validation is True:
            self.path = 'datasets/voxceleb1/validation/wav'
            self.speakers = glob.glob(''.join([self.path, '/*']))
            # Compute min length of utternaces per speaker
            # self.utterance_number = min([
            #     len(glob.glob(sp+'/*/*.wav')) for sp in self.speakers])
            # config.SPEAKER_M = self.utterance_number
            self.utterance_number = config.SPEAKER_M
            shuffle(self.speakers)
            return

        if test is True:
            # TEST
            self.path = 'datasets/voxceleb1/test/wav'
            self.speakers = glob.glob(''.join([self.path, '/*']))
            self.utterance_number = config.SPEAKER_M  # Test
            shuffle(self.speakers)
            return

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):

        speaker = self.speakers[idx]
        wav_files = glob.glob(speaker+'/*/*.wav')

        # At each call different samples of the same speaker
        # are given to the dataloader. Every epoch the model
        # pottentially learns different samples for the same speaker.
        shuffle(wav_files)
        wav_files = wav_files[0:self.utterance_number]

        mel_dbs = []
        for f in wav_files:
            _, mel_db, _ = mfccs_and_spec(f, wav_process=True)
            mel_dbs.append(mel_db)
        return torch.Tensor(mel_dbs)
