"""Create a test file for speaker verification using RAVDESS dataset"""
import itertools
import os
from random import shuffle
import numpy as np
from utils.load_dataset import load_RAVDESS


def export_verification_file(pairs=None, path='./temp.txt'):
    with open(file=path, mode="w") as file:
        for pair in pairs:
            label, u1, u2 = pair
            file.write(f"{label} {u1} {u2}\n")
    print(f'Export finished for {path} with {len(pairs)} pairs.')


# Load dataset where each label is a list containing
# [modality, vocal_channel, emotion,
# emotional_intensity, statement,
# repetition, actor]
X, y = load_RAVDESS(train_only=True, labels_only=False)

# Remove dataset path from sample paths
X = [x.replace("datasets/ravdess/", "") for x in X]
# Get labels casting string to int-1
# 0 = neutral, 1 = calm, 2 = happy, 3 = sad,
# 4 = angry, 5 = fearful, 6 = disgust, 7 = surprised
emotion = [int(l[2])-1 for l in y]
emotional_intensity = [int(l[3])-1 for l in y]  # 0 normal, 1 strong
speaker_id = [int(l[-1])-1 for l in y]  # 0,1,2,...,23 (12 men 12 women)
statement = [int(l[4])-1 for l in y]  # 0 or 1
repetition = [int(l[5])-1 for l in y]  # 0 or 1

# Variables aligned with matrix dimensions
SPEAKERS = 24
EMOTIONS = 8
STATEMENTS = 2
INTENSITIES = 2
REPETITIONS = 2


# Create an indexing list for emotions
idx = np.zeros((SPEAKERS, EMOTIONS, STATEMENTS,
                INTENSITIES, REPETITIONS), dtype=int)

# Inverse mapping
x_contains = [(speaker_id[i], emotion[i], statement[i], emotional_intensity[i], repetition[i])
              for i in range(len(X))]

# invert to get X[ idx[a,b,c,d] ] = sample
for i in range(len(X)):
    sp, em, st, inte, rep = x_contains[i]
    idx[sp, em, st, inte, rep] = i

# INITIALIZE
# neutral samples
X_neutral = []
# normal emotion samples
X_norm = []
# strong emotion samples
X_strong = []


# NORMAL EMOTION IN ENROLLEMENT UTTERANCE
# VERIFICATION UTTERANCE IS "EMOTION FREE"
# a list of tuples (label,utterance_1,utterance2)
# which we will evaluate during test time for verification
pairs_norm = []
# Outer product to reduce time
for sp, em, st, rep in itertools.product(range(SPEAKERS),
                                         range(1, EMOTIONS),
                                         range(STATEMENTS),
                                         range(REPETITIONS)):
    # Normal-emotionally enrollment utterance
    enrollment = X[idx[sp, em, st, 0, rep]]  # with emotion
    verification = X[idx[sp, 0, st, 0, rep]]  # no emotion, no intens
    # add pair with the same speaker, so label=1
    pairs_norm.append((1, enrollment, verification))

    # create a list without `sp` speaker id to pick from
    left_speakers = list(range(0, sp)) + list(range(sp+1, SPEAKERS))
    # add a different speaker (label=0)
    diff_sp = int(np.random.choice(left_speakers, 1))
    # Normal-emotionally enrollment utterance
    # but this time from another speaker
    enrollment = X[idx[diff_sp, em, st, 0, rep]]  # with emotion
    verification = X[idx[sp, 0, st, 0, rep]]  # no emotion,intens
    # same speaker so label=1
    pairs_norm.append((0, enrollment, verification))

pairs_strong = []
# Outer product to reduce time
for sp, em, st, rep in itertools.product(range(SPEAKERS),
                                         range(1, EMOTIONS),
                                         range(STATEMENTS),
                                         range(REPETITIONS)):
    # Normal-emotionally enrollment utterance
    enrollment = X[idx[sp, em, st, 1, rep]]  # with emotion
    verification = X[idx[sp, 0, st, 0, rep]]  # no emotion, no intens
    # add pair with the same speaker, so label=1
    pairs_strong.append((1, enrollment, verification))

    # create a list without `sp` speaker id to pick from
    left_speakers = list(range(0, sp)) + list(range(sp+1, SPEAKERS))
    # add a different speaker (label=0)
    diff_sp = int(np.random.choice(left_speakers, 1))
    # Normal-emotionally enrollment utterance
    # but this time from another speaker
    enrollment = X[idx[diff_sp, em, st, 1, rep]]  # with emotion
    verification = X[idx[sp, 0, st, 0, rep]]  # no emotion,intens
    # same speaker so label=1
    pairs_strong.append((0, enrollment, verification))

# shuffle
shuffle(pairs_norm)
shuffle(pairs_strong)
# Create path folder
os.makedirs('datasets/ravdess/veri_files/', exist_ok=True)
# Create a file as [labels, files1, files2]
export_verification_file(pairs=pairs_norm,
                         path="datasets/ravdess/veri_files/veri_test_exp1.1.txt")

export_verification_file(pairs=pairs_strong,
                         path="datasets/ravdess/veri_files/veri_test_exp1.2.txt")
