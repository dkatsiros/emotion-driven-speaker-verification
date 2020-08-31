from lib import sound_processing
import os


def read_file_identifiers(file=None, labels_only=True):
    """
    Reads filename and returns identifiers.
    """
    if file is None:
        raise FileNotFoundError()

    # Get filename only
    filename = os.path.basename(file)

    identifiers = filename.replace('.wav', '').split('-')

    # Return label only ?
    # 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    if labels_only is True:
        [modality, vocal_channel,
         emotion, emotional_intensity,
         statement, repetition, actor] = identifiers
        return int(emotion)

    return identifiers


def parse_wav(filename=None):
    """Return read file using librosa."""
    # Check file existance
    if filename is None:
        raise FileNotFoundError()
    loaded_file = sound_processing.load_wav(filename=filename)

    return loaded_file
