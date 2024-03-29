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
            self.speakers = X
            self.utterance_number = config.SPEAKER_M
            shuffle(self.speakers)
            return

        if validation is True:
            self.speakers = X
            self.utterance_number = config.SPEAKER_M
            shuffle(self.speakers)
            return

        if test is True:
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


class Voxceleb1_Evaluation_PreComputedMelSpectr(Dataset):
    def __init__(self, test_file_path='datasets/voxceleb1/test/veri_test2.txt',
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
        path_to_test = 'datasets/voxceleb1/test/wav'
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


class Voxceleb1PreComputedMelSpectr(Dataset):
    """Fast implementation of PyTorch's abstruct dataset for Voxceleb"""

    def __init__(self, X, training=False, validation=False, test=False,
                 max_seq_len=None, fixed_length=True, fe_method="MEL_SPECTROGRAM",
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
        wav_files = glob.glob(speaker+'/*/*.npy')
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
