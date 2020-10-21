import time
import numpy as np
import glob2 as glob
from tqdm import tqdm

from lib import sound_processing
from lib.sound_processing import load_wav, get_melspectrogram


class PreComputeMelSpectrograms():
    """Fast computing of mel spectrograms in python"""

    def __init__(self, paths, max_seq_len=245, *args, **kwargs):
        self.max_seq_len = max_seq_len
        self.fixed_length = True
        for path in paths:
            print(f"Loading {path}...")
            for file in tqdm(glob.glob(path)):
                mel_array = self.get_mel_spectrogram(file)
                # File that should be discourted / empty files
                if mel_array is None:
                    continue
                filepath = self.melname(file)
                self.save(mel_file=mel_array, path=filepath)

    def melname(self, wavname):
        """
        Create the new name of a given wavname
        example : "dataset/0001.wav" -> "dataset/0001_mel.png
        """
        x = wavname.split('.')[:-1]
        return ''.join([*x, '_mel.npy'])

    def save(self, mel_file, path):
        """Save the read by librosa in path"""
        assert type(mel_file).__module__ == np.__name__
        np.save(path, mel_file)
        print(path)
        exit()

    def get_mel_spectrogram(self, file):
        """Tries to read the file and return the mel spectrogram of it"""
        try:
            x = load_wav(file)
            x = get_melspectrogram(x)
            return x
        except:
            return None

    # def zero_pad_and_stack(self, X):
    #     """
    #     This function performs zero padding on a list of features and forms them into a numpy 3D array

    #     Returns:
    #         padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
    #     """

    #     max_length = self.max_seq_len

    #     feature_dim = X[0].shape[-1]
    #     padded = np.zeros((max_length, feature_dim))

    #     # Do the actual work

    #     if X.shape[0] < max_length:
    #         # Needs padding
    #         diff = max_length - X.shape[0]
    #         # pad
    #         X = np.vstack((X, np.zeros((diff, feature_dim))))
    #     else:
    #         if self.fixed_length is True:
    #             # Set a fixed length => information loss
    #             X = np.take(X, list(range(0, max_length)), axis=0)
    #     # Add to padded
    #     return X
