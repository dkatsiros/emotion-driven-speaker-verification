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
        return filenames, labels


def emotion2idx(emotion=None):
    """Get an emotion as labeled originally and return a mapping."""
    if emotion is None:
        raise AssertionError
    mapping = {
        # Neutral
        'neu': 0,
        # Anger
        'ang': 1,
        # Joy
        'hap': 2,
        # Sadness
        'sad': 3,
        # Frustrated
        'fru': 4,
        # Excited,
        'exc': 5,
        # Surprised
        'sur': 6,
        # Fear
        'fea': 7,
        # Disgust
        'dis': 8,
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


def get_classes(n_classes=9):
    if n_classes == 4:
        return ['Neutral', 'Anger', 'Happiness', 'Sadness']
    elif n_classes == 6:
        return ['Neutral', 'Anger', 'Happiness', 'Sadness',
                'Frustrated', 'Excited']
    elif n_classes == 9:
        return ['Neutral', 'Anger', 'Happiness',
                'Sadness', 'Frustrated', 'Excited',
                'Surprised', 'Fear', 'Disgust']


def get_categories_population_dictionary(labels, n_classes=9):
    """Return a mapping (category) -> Population."""
    mapping = {i: 0 for i in range(0, n_classes)}
    # Iterate each file and map
    for l in labels:
        if l >= n_classes:
            continue
        mapping[l] += 1
    return mapping


def parse_wav(filename=None):
    """Return read file using librosa."""
    # Check file existance
    if filename is None:
        raise FileNotFoundError()
    loaded_file = sound_processing.load_wav(filename=filename)

    return loaded_file
