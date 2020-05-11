import os
import numpy as np

def load_word_vectors(file, dim=50):
    """Read the word vectors from a file.

    Arguments:
        file {str} -- the filename
        dim {int} -- the dimension of the embeddings

    Returns:
        word2idx {dict} -- dictionary of words to ids
        idx2word {dict} -- dictionary of ids to words
        embeddings {numpy.ndarray} -- the word embeddings matrix
    """
    if not os.path.exists(file):
        raise FileNotFoundError

    # Create empty dictionaries and list
    word2idx = {}
    idx2word = {}
    embeddings = []

    # Reserve idx = 0 for zero padding
    embeddings.append(np.zeros(dim))

    # Read file
    with open(file, mode='r', encoding='utf-8')as f:
        # Process each line
        for idx, line in enumerate(f, 1):
            # Split line
            values = line.rstrip().split(" ")
            # Split values
            word = values[0]
            vector = np.array(values[1], dtype='float32')

            # Convert to idx
            word2idx[word] = idx
            idx2word[idx] = word
            embeddings.append(vector)

            # Check for unknown
            if '<unk>' not in word2idx:
                word2idx['<unk>'] = len(word2idx) + 1
                idx2word[len(idx2word) + 1] = '<unk>'
                embeddings.append(
                    np.random.uniform(low=-0.05, high=0.05, size=dim)
                )

        print(f'Loaded {len(embeddings)} word vectors')

        return word2idx, idx2word, embeddings