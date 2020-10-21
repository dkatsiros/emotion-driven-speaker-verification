import time
import numpy as np
import glob2 as glob
from tqdm import tqdm

from lib import sound_processing
from lib.sound_processing import load_wav, get_melspectrogram


class PreComputeMelSpectrograms():
    """Fast computing of mel spectrograms in python"""

    def __init__(self, paths, *args, **kwargs):
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

    def get_mel_spectrogram(self, file):
        """Tries to read the file and return the mel spectrogram of it"""
        try:
            x = load_wav(file)
            x = get_melspectrogram(x)
            return x
        except:
            return None
