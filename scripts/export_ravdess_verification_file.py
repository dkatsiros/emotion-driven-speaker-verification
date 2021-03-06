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

#####################
# EXPERIMENT 1
#####################

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
    # same speaker so label=0
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
    # same speaker so label=0
    pairs_strong.append((0, enrollment, verification))

#####################
# RUN EXPERIMENT 1
#####################

# # shuffle
# shuffle(pairs_norm)
# shuffle(pairs_strong)
# # Create path folder
# os.makedirs('datasets/ravdess/veri_files/', exist_ok=True)
# # Create a file as [labels, files1, files2]
# export_verification_file(pairs=pairs_norm,
#                          path="datasets/ravdess/veri_files/veri_test_exp1.1.txt")

# export_verification_file(pairs=pairs_strong,
#                          path="datasets/ravdess/veri_files/veri_test_exp1.2.txt")


########################
# EXPERIMENT 2
########################

# NORMAL EMOTION IN ENROLLEMENT UTTERANCE
# VERIFICATION UTTERANCE IS "EMOTION FREE"
# a list of tuples (label,utterance_1,utterance2)
# which we will evaluate during test time for verification
pairs_norm = []
# Outer product to reduce time
for sp, em, em2, st, rep in itertools.product(range(SPEAKERS),
                                              range(1, EMOTIONS),
                                              range(1, EMOTIONS),
                                              range(STATEMENTS),
                                              range(REPETITIONS)):
    # take all the emotions over the diagonal
    # of the matrix EMOTIONS @ EMOTIONS
    if em > em2:
        # Same speaker
        enrollment = X[idx[sp, em, st, 0, rep]]  # with emotion
        verification = X[idx[sp, em2, st, 0, rep]]  # no emotion, no intens
        # add pair with the same speaker, so label=1
        pairs_norm.append((1, enrollment, verification))

        # create a list without `sp` speaker id to pick from
        left_speakers = list(range(0, sp)) + list(range(sp+1, SPEAKERS))
        # add a different speaker (label=0)
        diff_sp = int(np.random.choice(left_speakers, 1))
        # Normal-emotionally enrollment utterance
        # but this time from another speaker
        enrollment = X[idx[diff_sp, em, st, 0, rep]]  # with emotion
        verification = X[idx[sp, em2, st, 0, rep]]  # no emotion,intens
        # same speaker so label=0
        pairs_norm.append((0, enrollment, verification))

pairs_strong = []
# Outer product to reduce time
for sp, em, em2, st, rep in itertools.product(range(SPEAKERS),
                                              range(1, EMOTIONS),
                                              range(1, EMOTIONS),
                                              range(STATEMENTS),
                                              range(REPETITIONS)):
    # take all the emotions over the diagonal
    # of the matrix EMOTIONS @ EMOTIONS
    if em > em2:
        # Same speaker
        enrollment = X[idx[sp, em, st, 1, rep]]  # high intensity
        verification = X[idx[sp, em2, st, 1, rep]]  # high intensity
        # add pair with the same speaker, so label=1
        pairs_strong.append((1, enrollment, verification))

        # create a list without `sp` speaker id to pick from
        left_speakers = list(range(0, sp)) + list(range(sp+1, SPEAKERS))
        # add a different speaker (label=0)
        diff_sp = int(np.random.choice(left_speakers, 1))
        # but this time from another speaker
        enrollment = X[idx[diff_sp, em, st, 1, rep]]  # high intensity
        verification = X[idx[sp, em2, st, 1, rep]]  # high intensity
        # diff speaker => label=0
        pairs_strong.append((0, enrollment, verification))

#####################
# RUN EXPERIMENT 2
#####################

# # shuffle
# shuffle(pairs_norm)
# shuffle(pairs_strong)
# # Create path folder
# os.makedirs('datasets/ravdess/veri_files/', exist_ok=True)
# # Create a file as [labels, files1, files2]
# export_verification_file(pairs=pairs_norm,
#                          path="datasets/ravdess/veri_files/veri_test_exp2.1.txt")

# export_verification_file(pairs=pairs_strong,
#                          path="datasets/ravdess/veri_files/veri_test_exp2.2.txt")

#####################
# EXPERIMENT 3
#####################

# NORMAL EMOTION IN ENROLLEMENT UTTERANCE
# VERIFICATION UTTERANCE IS "EMOTION FREE"
# a list of lists of tuples (label,utterance_1,utterance2)
# which we will evaluate during test time for verification
# each list this time contains a different emotion on verification
# while we keep enrollment utterance neutral
pairs_norm = [[] for i in range(EMOTIONS-1)]
# Outer product to reduce time
for sp, em, st, rep in itertools.product(range(SPEAKERS),
                                         range(1, EMOTIONS),
                                         range(STATEMENTS),
                                         range(REPETITIONS)):
    # Normal-emotionally enrollment utterance
    enrollment = X[idx[sp, em, st, 0, rep]]  # with emotion
    verification = X[idx[sp, 0, st, 0, rep]]  # no emotion, no intens
    # add pair with the same speaker, so label=1
    pairs_norm[em-1].append((1, enrollment, verification))

    # create a list without `sp` speaker id to pick from
    left_speakers = list(range(0, sp)) + list(range(sp+1, SPEAKERS))
    # add a different speaker (label=0)
    diff_sp = int(np.random.choice(left_speakers, 1))
    # Normal-emotionally enrollment utterance
    # but this time from another speaker
    enrollment = X[idx[diff_sp, em, st, 0, rep]]  # with emotion
    verification = X[idx[sp, 0, st, 0, rep]]  # no emotion,intens
    # same speaker so label=0
    pairs_norm[em-1].append((0, enrollment, verification))

pairs_strong = [[] for i in range(EMOTIONS-1)]
# Outer product to reduce time
for sp, em, st, rep in itertools.product(range(SPEAKERS),
                                         range(1, EMOTIONS),
                                         range(STATEMENTS),
                                         range(REPETITIONS)):
    # Normal-emotionally enrollment utterance
    enrollment = X[idx[sp, em, st, 1, rep]]  # with emotion
    verification = X[idx[sp, 0, st, 0, rep]]  # no emotion, no intens
    # add pair with the same speaker, so label=1
    pairs_strong[em-1].append((1, enrollment, verification))

    # create a list without `sp` speaker id to pick from
    left_speakers = list(range(0, sp)) + list(range(sp+1, SPEAKERS))
    # add a different speaker (label=0)
    diff_sp = int(np.random.choice(left_speakers, 1))
    # Normal-emotionally enrollment utterance
    # but this time from another speaker
    enrollment = X[idx[diff_sp, em, st, 1, rep]]  # with emotion
    verification = X[idx[sp, 0, st, 0, rep]]  # no emotion,intens
    # same speaker so label=0
    pairs_strong[em-1].append((0, enrollment, verification))

#####################
# RUN EXPERIMENT 3
#####################

# # shuffle
# shuffle(pairs_norm)
# shuffle(pairs_strong)
# # Create path folder
# os.makedirs('datasets/ravdess/veri_files/', exist_ok=True)
# # Create a file as [labels, files1, files2]
# for emotion, pairs in enumerate(pairs_norm, 1):
#     shuffle(pairs)
#     export_verification_file(pairs=pairs,
#                              path=f"datasets/ravdess/veri_files/veri_test_exp3.1.{emotion}.txt")
# for emotion, pairs in enumerate(pairs_strong, 1):
#     shuffle(pairs)
#     export_verification_file(pairs=pairs,
#                              path=f"datasets/ravdess/veri_files/veri_test_exp3.2.{emotion}.txt")


# EXPERIMENT 4
#####################

# Does emotion ignorance affect EER ?
# In order to identify this we create two
# different verification files.
# File 1 (emotional_ignorance):
# each pair should be a neutral (emotion-free)
# enrollment while verification utterance
# should be emotion
# File 2 (emotional_knowledge):
# each pair should be a emotional enrollment
# as well as an emotional verification utterance
# Files are meant to be:
# a list of lists of tuples (label,utterance_1,utterance2)
# which we will evaluate during test time for verification
emotional_ignorance = [[] for i in range(EMOTIONS-1)]
# Outer product to reduce time
for sp, intensity, em, st, rep in itertools.product(range(SPEAKERS),
                                                    # both intensities
                                                    range(INTENSITIES),
                                                    range(1, EMOTIONS),
                                                    range(STATEMENTS),
                                                    range(REPETITIONS)):
    # Emotional Ignorance  emotion
    enrollment = X[idx[sp, 0, st, 0, rep]]  # no emotion, no intens
    verification = X[idx[sp, em, st, intensity, rep]]  # with emotion
    # add pair with the same speaker, so label=1
    emotional_ignorance[em-1].append((1, enrollment, verification))

    # create a list without `sp` speaker id to pick from
    left_speakers = list(range(0, sp)) + list(range(sp+1, SPEAKERS))
    # add a different speaker (label=0)
    diff_sp = int(np.random.choice(left_speakers, 1))
    # Normal-emotionally enrollment utterance
    # but this time from another speaker
    enrollment = X[idx[sp, 0, st, intensity, rep]]  # no emotion,intens
    verification = X[idx[diff_sp, em, st, 0, rep]]  # with emotion
    # same speaker so label=0
    emotional_ignorance[em-1].append((0, enrollment, verification))


emotional_knowledge = [[] for i in range(EMOTIONS-1)]
# Outer product to reduce time
for (sp, intensity, em, st, rep,
     intensity2, st2, rep2
     ) in itertools.product(range(SPEAKERS),
                            range(INTENSITIES),  # both intensities
                            range(1, EMOTIONS),
                            range(STATEMENTS),
                            range(REPETITIONS),
                            # second utterance iteration
                            range(INTENSITIES),
                            range(STATEMENTS),
                            range(REPETITIONS)):
    encoding_1 = intensity + 2*st + 4*rep
    encoding_2 = intensity2 + 2*st2 + 4*rep2
    # create 8x8 matrix with all possible combinations
    # (encoding1 @ encoding2) and get all elements over
    # the diagonal line (not dublicate + not same enroll+verif utt)
    if not (encoding_1 > encoding_2):
        continue
    # Emotional knowledge  emotion
    enrollment = X[idx[sp, em, st2, intensity2, rep2]]  # with emotion
    verification = X[idx[sp, em, st, intensity, rep]]  # with emotion
    # add pair with the same speaker, so label=1
    emotional_knowledge[em-1].append((1, enrollment, verification))

    # create a list without `sp` speaker id to pick from
    left_speakers = list(range(0, sp)) + list(range(sp+1, SPEAKERS))
    # add a different speaker (label=0)
    diff_sp = int(np.random.choice(left_speakers, 1))
    # Normal-emotionally enrollment utterance
    # but this time from another speaker
    # enrollment = X[idx[sp, 0, st, intensity, rep]]  # with emotion
    enrollment = X[idx[sp, 0, st2, intensity2, rep2]]  # with emotion
    verification = X[idx[diff_sp, em, st, intensity, rep]]  # with emotion
    # different speaker so label=0
    emotional_knowledge[em-1].append((0, enrollment, verification))

# Create path folder
os.makedirs('datasets/ravdess/veri_files/', exist_ok=True)
# Create a file as [labels, files1, files2]
for emotion, pairs in enumerate(emotional_ignorance, 1):
    shuffle(pairs)
    export_verification_file(pairs=pairs,
                             path=f"datasets/ravdess/veri_files/veri_test_exp4.1.{emotion}.txt")
for emotion, pairs in enumerate(emotional_knowledge, 1):
    shuffle(pairs)
    export_verification_file(pairs=pairs,
                             path=f"datasets/ravdess/veri_files/veri_test_exp4.2.{emotion}.txt")
