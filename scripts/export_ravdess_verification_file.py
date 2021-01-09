"""Create a test file for speaker verification using RAVDESS dataset"""
from random import shuffle
from utils.load_dataset import load_RAVDESS


def export_verification_file(pairs=None, path='./temp.txt'):
    with open(file=path, mode="w") as file:
        for pair in pairs:
            label, u1, u2 = pair
        file.write(f"{label} {u1} {u2}\n")


# Load dataset where each label is a list containing
# [modality, vocal_channel, emotion,
# emotional_intensity, statement,
# repetition, actor]
X, y = load_RAVDESS(train_only=True, labels_only=False)

# Get labels casting string to int-1
# 0 = neutral, 1 = calm, 2 = happy, 3 = sad,
# 4 = angry, 5 = fearful, 6 = disgust, 7 = surprised
emotion = [int(l[2])-1 for l in y]
emotional_intensity = [int(l[3])-1 for l in y]  # 0 normal, 1 strong
speaker_id = [int(l[-1])-1 for l in y]  # 0 to 23 (12 men 12 women)

# INITIALIZE
# neutral samples
X_neutral = []
speaker_id_neutral = []
# normal emotion samples
X_norm = []
speaker_id_norm = []
emotion_norm = []
# strong emotion samples
X_strong = []
speaker_id_strong = []
emotion_strong = []

for x, emotion, intensity, spkr in zip(X, emotion, emotional_intensity, speaker_id):
    # neutral has no intensity
    if emotion == 0:
        X_neutral.append(x)
        speaker_id_neutral.append(spkr)
    else:
        # normal intensity
        if intensity == 0:
            X_norm.append(x)
            speaker_id_norm.append(spkr)
            emotion_norm.append(emotion)
        # strong intensity
        else:
            X_strong.append(x)
            speaker_id_strong.append(spkr)
            emotion_strong.append(emotion)

# NORMAL EMOTION IN ENROLLEMENT UTTERANCE
# VERIFICATION UTTERANCE IS "EMOTION FREE"
# a list of tuples (label,utterance_1,utterance2)
# which we will evaluate during test time for verification
pairs_norm = []
for idx_neutral in range(len(X_neutral)):
    for idx_norm in range(len(X_norm)):
        # Get label for speaker verification if same speakers
        if speaker_id_norm[idx_norm] == speaker_id_neutral[idx_neutral]:
            label = 1
        else:
            label = 0
        # actual samples
        u1 = X_norm[idx_norm]
        u2 = X_neutral[idx_neutral]
        # save pairs for exporting into file
        pair = (label, u1, u2)
        pairs_norm.append(pair)

# STRONG EMOTION IN ENROLLMENT UTTERANCE
# VERIFICATION UTTERANCE IS "EMOTION FREE"
# a list of tuples (label,utterance_1,utterance2)
# which we will evaluate during test time for verification
pairs_strong = []
for idx_neutral in range(len(X_neutral)):
    for idx_strong in range(len(X_strong)):
        # Get label for speaker verification if same speakers
        if speaker_id_strong[idx_strong] == speaker_id_neutral[idx_neutral]:
            label = 1
        else:
            label = 0
        # enrollment utterance
        u1 = X_strong[idx_strong]
        # verification utterance
        u2 = X_neutral[idx_neutral]
        # save pairs for exporting into file
        pair = (label, u1, u2)
        pairs_strong.append(pair)

# shuffle
shuffle(pairs_norm)
shuffle(pairs_strong)
# Create a file as [labels, files1, files2]
export_verification_file(pairs=pairs_norm,
                         path="datasets/ravdess/veri_test_exp1.1.txt")

export_verification_file(pairs=pairs_norm,
                         path="datasets/ravdess/veri_test_exp1.2.txt")
