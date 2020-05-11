import os
from config import EMB_PATH, EMB_DIM, EMB_FILE
from utils.load_embeddings import load_word_vectors


EMBEDDINGS = os.path.join(EMB_PATH, EMB_FILE)

word2idx, idx2word, embeddings = load_word_vectors(file=EMBEDDINGS, dim=EMB_DIM)