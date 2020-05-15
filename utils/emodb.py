from utils import sound_processing

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
    loaded_file = sound_processing.load_wav(filename=filename)
    # Get file name details
    speaker, phrase, emotion, version = get_file_details(filename)
    return [loaded_file, int(speaker), phrase, emotion2idx(emotion), version]


def get_indexes_for_wav_categories(files):
    """Return a mapping (category) -> (indexes)."""
    mapping = {i:[] for i in range(0,7)}
    # Iterate each file and map
    for idx, file in enumerate(files):
        emotion_idx = file[3]
        mapping[emotion_idx].append(idx)
    return mapping

