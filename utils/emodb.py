import librosa
from config import SAMPLING_RATE

def get_file_details(filename):
    """Split the filename, return the details."""
    import re
    match = re.search(r'.*/(.+?).wav',filename)
    if not match:
        return None
    # Regex match
    file = match.group(1)
    # file = filename.split('/')[-1:]
    # file = file[0]
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


def parse_wav(filename=None):
    """Return read file using librosa."""
    # Check file existance
    if filename is None:
        return None
    # Load file using librosa
    loaded_file = librosa.load(filename, sr=SAMPLING_RATE)
    # Get file name details
    speaker, phrase, emotion, version = get_file_details(filename)
    return [loaded_file, int(speaker), phrase, emotion2idx(emotion), version]
