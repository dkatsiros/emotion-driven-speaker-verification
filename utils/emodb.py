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
    return filename, speaker, phrase, emotion, version

def load_emotions_mapping():
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
    return mapping