import librosa
import numpy as np
from config import SAMPLING_RATE, WINDOW_LENGTH, HOP_LENGTH

def get_file_details(filename):
    """Split the filename, return the details."""
    import re
    match = re.search(r'.*/(.+?).wav',filename)
    if not match:
        return None
    # Regex match
    file = match.group(1)
    # Speaker
    speaker = file[:2]
    # Phrase
    phrase = file[2:5]
    # Emotion
    emotion = file[5]
    # Different attempts from the same speaker, on the same utterance
    # and the same emotion.
    version = file[6]
    return speaker, phrase, emotion, version


def emotion2idx(emotion=None):
    """Get an emotion in German and return a mapping."""
    if emotion is None:
        raise AssertionError
    mapping = {
        # Neutral
        'N': 0,
        # Anger
        'W': 1,
        # Fear
        'A': 2,
        # Joy
        'F': 3,
        # Sadness
        'T': 4,
        # Disgust
        'E': 5,
        # Boredom
        'L': 6
    }
    return mapping[emotion]


def idx2emotion(idx=None):
    """Return emotion name in English."""
    if idx is None:
        return None
    # Create mapping
    mapping = {
        0: 'Neutral',
        1: 'Anger',
        2: 'Fear',
        3: 'Joy',
        4: 'Sadness',
        5: 'Disgust',
        6: 'Boredom'
    }
    return mapping[idx]


def parse_wav(filename=None):
    """Return read file using librosa."""
    # Check file existance
    if filename is None:
        return None
    # Load file using librosa
    loaded_file, _ = librosa.load(filename, sr=SAMPLING_RATE)
    # Get file name details
    speaker, phrase, emotion, version = get_file_details(filename)
    return [loaded_file, int(speaker), phrase, emotion2idx(emotion), version]


def get_mfcc(wav, sr=SAMPLING_RATE):
    return librosa.feature.mfcc(wav, sr, n_mfcc=13, win_length=WINDOW_LENGTH,
                                hop_length=HOP_LENGTH)


def get_mfcc_with_deltas(wav, sr=SAMPLING_RATE):
    """Return MFCC, delta and delta-delta of a given wav."""
    mfcc = get_mfcc(wav, sr)
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta_delta = librosa.feature.delta(mfcc,order=2)
    return np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta))


def get_features_mean_var(loaded_wav=None):
    """Compute features from a loaded file. Computes MFCC
    coefficients and then the mean and variance along all frames
    for each coefficient.

    Keyword Arguments:
        loaded_wav {np.array} -- Wav read using librosa (default: {None})

    Returns:
        np.array -- An array containing mean and variance for each one of
                    the MFCC features as (m1,v1,m2,v2, ..) across all frames.
    """
    # Check inputs
    if loaded_wav is None:
        return None
    # Compute mfcc, delta and delta delta
    mfccs = get_mfcc_with_deltas(loaded_wav)  # (39,#frames)
    # Compute mean along 1-axis
    mean = np.mean(mfccs, axis=1)  # (39,1)
    # Compute variance along 1-axis
    variance = np.var(mfccs, axis=1)  # (39,1)
    # Export the features - order 'F' to preserve (m1,v1,m2,v2, ... )
    return np.ravel((mean, variance), order='F')  # (78,) -- 1darray



def get_indexes_for_wav_categories(files):
    """Return a mapping (category) -> (indexes)."""
    mapping = {i:[] for i in range(0,7)}
    # Iterate each file and map
    for idx, file in enumerate(files):
        emotion_idx = file[3]
        mapping[emotion_idx].append(idx)
    return mapping

