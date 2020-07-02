from utils import sound_processing


def read_evaluation_file(eval_file=None):
    """
    Reads an evaluation file and returns two lists.
    The first list contains all valid file names and
    the second one the labels of each one.
    """
    if eval_file is None:
        raise FileNotFoundError()

    filenames = []
    labels = []
    with open(file=eval_file, mode='r') as f:

        # Read raw data, except two first lines
        data_raw = [line for line in f.readlines()][1:]
        data = []
        for idx, line in enumerate(data_raw):
            if line == "\n" and idx + 1 < len(data_raw):
                # Get the next line
                data.append(data_raw[idx+1])

        # For each file get name and label
        for line in data:
            split = line.split("\t")
            label = split[2]
            # If no label skip. Do not add anything
            if label == "xxx" or label == "oth":
                continue
            filenames.append(split[1])
            labels.append(emotion2idx(label))
        print(len(filenames), len(labels))
        return filenames, labels
        # wav_name =


def emotion2idx(emotion=None):
    """Get an emotion in German and return a mapping."""
    if emotion is None:
        raise AssertionError
    mapping = {
        # Neutral
        'neu': 0,
        # Anger
        'ang': 1,
        # Fear
        'fea': 2,
        # Joy
        'hap': 3,
        # Sadness
        'sad': 4,
        # Disgust
        'dis': 5,
        # Surprised
        'sur': 6,
        # Frustrated
        'fru': 7,
        # Excited,
        'exc': 8
    }
    return mapping[emotion]


def idx2emotion(idx=None):
    """Return emotion name in English."""
    if idx is None:
        raise ValueError
    # Get classes
    classes = get_classes()
    # Create mapping
    mapping = {i: c for i, c in enumerate(classes)}
    return mapping[idx]


def get_classes():
    return ['Neutral', 'Anger', 'Fear',
            'Happiness', 'Sadness', 'Disgust',
            'Surprised', 'Frustrated', 'Excited']
