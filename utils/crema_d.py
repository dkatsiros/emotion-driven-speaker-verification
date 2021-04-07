import os
import numpy as np
import csv
import copy
import itertools

EXPORTED_FOLDER = "emotional_speaker_verification_exported"


##########################
# Experiment Setup Scripts
##########################

def export_verification_file(pairs=None,
                             filepath='datasets/crema-d/emotional_speaker_verification_exported/veri.txt',
                             override=False):

    if os.path.exists(filepath) is True and override is False:
        print(f"File {filepath} already exists. Aborting export..")
        return

    with open(file=filepath, mode="w") as file:
        for pair in pairs:
            label, u1, u2 = pair
            file.write(f"{label} {u1} {u2}\n")
    print(f'Export finished for {filepath} with {len(pairs)} pairs.')


def export_csv(tuples, filepath=None, override=False):
    """Export a csv file, for constant arrangement in files."""
    # Do not overwrite each time
    if os.path.exists(filepath) is True and override is False:
        print(f"File {filepath} already exists. Aborting export..")
        return
    # If no-file, export
    with open(filepath, mode="w") as file:
        writer = csv.writer(file,
                            delimiter=",",
                            quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        # Headers
        writer.writerow(["CLIPNAME", "SPEAKER_ID", "SEX",
                         "EMOTION", "INTENSITY", "SENTENCE"])
        # Actual exporting
        for row in tuples:
            writer.writerow([c for c in row])
    print(f'Export finished for {filepath}.')


def import_csv(filepath=None):
    tuples = []
    assert os.path.exists(filepath)
    with open(filepath, mode="r") as file:
        csv_reader = csv.reader(file,
                                delimiter=",",
                                quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue
            tuples.append([row[0],
                           int(row[1]),
                           int(row[2]),
                           int(row[3]),
                           float(row[4]),
                           int(row[5])])
    return tuples


def read_label_file(dataset_path=None, filename="summaryTable.csv", source=0):
    """source:
        0 -> audio
        1 -> visual
        2 -> audiovisual
"""

    if dataset_path is None:
        raise AssertionError()
    ##
    # pick one of 12 spoken sentences available
    sentences = ["IEO", "TIE", "IOM", "IWW", "TAI",
                 "MTI", "IWL", "ITH", "DFA", "ITS",
                 "TSI", "WSI"]
    # map to int
    sentences_dict = dict([(y, x+1)
                           for x, y in enumerate(sorted(set(sentences)))])

    ##
    # emotion to index mapping
    emotions = ["N", "A", "H", "S", "D", "F"]
    emot2idx = {e: i for i, e in enumerate(emotions)}

    ##
    # get sex dictionary
    sex_dict = {}
    csv_file = os.path.join(dataset_path, "VideoDemographics.csv")
    with open(csv_file, mode="r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue
            # get demographic for idx's speaker
            speaker_id = int(row[0]) - 1000  # mapped to [1-91]
            age = row[1]
            sex = row[2]
            race = row[3]
            ethnicity = row[4]
            # save for later
            sex_dict[speaker_id] = 1 if sex == "Male" else 0
    ##
    # export the final file
    csv_file = os.path.join(dataset_path, "processedResults", filename)
    # export metadata in a variable with tuples
    files_labels = []
    with open(csv_file, mode="r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for idx, row in enumerate(csv_reader):
            # header
            if idx == 0:
                continue
            # Extract information
            clip_name = row[1]
            # split filename and map to data
            speaker_id = int(clip_name.split("_")[0]) - 1000
            sentence_num = sentences_dict[clip_name.split("_")[1]]
            # labels
            label_intensity_list = [float(i.strip())
                                    for i in row[3+2*source].split(":")]
            label_intensity_idx = np.argmax(label_intensity_list)
            label_intensity = label_intensity_list[label_intensity_idx]
            label = row[2+2*source].split(":")[label_intensity_idx]

            # create tuple
            files_labels.append((clip_name,  # actual filename
                                 speaker_id,  # 1-91
                                 sex_dict[speaker_id],  # 1 if male else 0
                                 emot2idx[label],  # A,D,F,H,N,S
                                 label_intensity,  # 0-100
                                 sentence_num  # 1-12
                                 ))

    return files_labels


def load_SER_export_train_val_test(n_emotions=6,
                                   dataset_path="datasets/crema-d",
                                   SER_file='SER_files.csv',
                                   train_val_test_ratio=[0.85, 0.1, 0.15],
                                   override=False):
    filepath_SER = os.path.join(dataset_path, EXPORTED_FOLDER, SER_file)

    if not os.path.exists(filepath_SER):
        raise FileNotFoundError

    # decleare emotions
    emotions = ["N", "A", "H", "S", "D", "F"][:n_emotions]
    # load pairs
    SER_pairs = import_csv(filepath=filepath_SER)
    # discard some emotion pairs
    SER_pairs = [pair for pair in SER_pairs if pair[3] < n_emotions]
    emotion_dict = {i: 0 for i in range(n_emotions)}
    for pair in SER_pairs:
        emotion_dict[pair[3]] += 1

    total = sum(emotion_dict.values())
    print(f"Total utterances: {total}")
    print('Percentages per emotion:\n-------------------------------------')
    print(*[em + '     ' for em in emotions])
    print(*[str(v/total*100)[:5] + '%' for v in emotion_dict.values()])

    # Get ratio from input
    train_ratio, val_ratio, test_ratio = train_val_test_ratio
    # Split samples according to ratio
    print(int(np.ceil(train_ratio * len(SER_pairs))))
    train_val = SER_pairs[0:int(np.ceil(train_ratio * len(SER_pairs)))]
    test = SER_pairs[int(np.ceil(train_ratio * len(SER_pairs))):]

    train = train_val[int(np.ceil(val_ratio * len(train_val))):]
    val = train_val[0:int(np.ceil(val_ratio * len(train_val)))]
    print(
        f"Train samples: {len(train)}, Test samples: {len(test)}, Validation samples: {len(val)} ")

    # train
    filepath_SER_train = os.path.join(
        dataset_path, EXPORTED_FOLDER, SER_file[:-4] + 'train' + SER_file[-4:])
    export_csv(tuples=train, filepath=filepath_SER_train, override=override)

    # test
    filepath_SER_test = os.path.join(
        dataset_path, EXPORTED_FOLDER, SER_file[:-4] + 'test' + SER_file[-4:])
    export_csv(tuples=test, filepath=filepath_SER_test, override=override)

    # validation
    filepath_SER_val = os.path.join(
        dataset_path, EXPORTED_FOLDER, SER_file[:-4] + 'val' + SER_file[-4:])
    export_csv(tuples=val, filepath=filepath_SER_val, override=override)


def split_SER_SV(pairs=[],
                 n_emotions=6,
                 dataset_path="datasets/crema-d",
                 SER_file='SER_files.csv',
                 SV_file='SV_files.csv',
                 override=False):
    """Load files with labels and slit to male-female to avoid
    trivial solutions. Export splitted files and verification file.
    Then export emotion train/test/val to seperate files, while keeping
    only the number of emotions given."""

    import random

    # Check existance
    if not os.path.exists(os.path.join(dataset_path, EXPORTED_FOLDER)):
        os.makedirs(os.path.join(dataset_path, EXPORTED_FOLDER))

    # balance men and women
    males = list(set([t[1] for t in pairs if t[2] == 1]))
    females = list(set([t[1] for t in pairs if t[2] == 0]))
    random.shuffle(males)
    random.shuffle(females)

    # unify to people after balance
    SER_people = males[:24] + females[:21]
    SV_people = males[24:] + females[21:]

    # First, split to SER and SV
    SER_samples = [t for t in pairs if t[1] in SER_people]
    SV_samples = [t for t in pairs if t[1] in SV_people]
    random.shuffle(SER_samples)
    random.shuffle(SV_samples)

    ##
    # Speech Emotion Recognition
    # try to export csv, if not already there
    filepath_SER = os.path.join(dataset_path, EXPORTED_FOLDER, SER_file)
    export_csv(tuples=SER_samples, filepath=filepath_SER, override=override)

    ##
    # Speaker Verification
    # try to export csv, if not already there
    filepath_SV = os.path.join(dataset_path, EXPORTED_FOLDER, SV_file)
    export_csv(tuples=SV_samples, filepath=filepath_SV, override=override)

    ##
    # Train test split for SV
    train_SV_people = SV_people[:32]
    train_SV = [t for t in pairs if t[1] in train_SV_people]

    valid_SV_people = SV_people[32:38]
    valid_SV = [t for t in pairs if t[1] in valid_SV_people]

    test_SV_people = SV_people[38:]
    test_SV = [t for t in pairs if t[1] in test_SV_people]

    filepath_SV_train = os.path.join(
        dataset_path, EXPORTED_FOLDER, SV_file[:-4]+'_train'+SV_file[-4:])
    export_csv(tuples=train_SV, filepath=filepath_SV_train, override=override)

    filepath_SV_valid = os.path.join(
        dataset_path, EXPORTED_FOLDER, SV_file[:-4]+'_valid'+SV_file[-4:])
    export_csv(tuples=valid_SV, filepath=filepath_SV_valid, override=override)

    filepath_SV_test = os.path.join(
        dataset_path, EXPORTED_FOLDER, SV_file[:-4]+'_test'+SV_file[-4:])
    export_csv(tuples=test_SV, filepath=filepath_SV_test, override=override)

    ##
    # create test pairs
    veri_couples = []
    positive_couples = []
    negative_couples = []
    for idx1, idx2 in itertools.product(range(len(test_SV)),
                                        range(len(test_SV))):
        if idx1 <= idx2:
            continue

        sample_1 = test_SV[idx1][0]
        sample_2 = test_SV[idx2][0]
        # same_speaker = 1 if test_SV[idx1][1] == test_SV[idx2][1] else 0
        # veri_couples.append([same_speaker, sample_1, sample_2])
        # random.shuffle(veri_couples)
        if test_SV[idx1][1] == test_SV[idx2][1]:
            positive_couples.append([1, sample_1, sample_2])
        else:
            negative_couples.append([0, sample_1, sample_2])

    # Balance Potives/Negative Test
    number_of_positives = len(positive_couples)
    random.shuffle(positive_couples)
    random.shuffle(negative_couples)
    negative_couples = negative_couples[:number_of_positives]
    for positive, negative in zip(positive_couples, negative_couples):
        veri_couples.append(positive)
        veri_couples.append(negative)
    #
    # shuffle and export to file
    export_verification_file(pairs=veri_couples, override=override)

    # Export SER data to files to ensure that
    # train/val/test remains constant.
    # Also discard emotions greater than n_emotions.
    load_SER_export_train_val_test(n_emotions=n_emotions,
                                   dataset_path=dataset_path,
                                   SER_file=SER_file,
                                   train_val_test_ratio=[0.85, 0.1, 0.15],
                                   override=override)


############################
# Loading and handling utils
############################

def load_CREMAD_SER(n_emotions=4,
                    dataset_path="datasets/crema-d",
                    SER_file='SER_files.csv'):
    """Return training, validation and testing (files,labels)"""
    # create filenames
    train_filepath = os.path.join(dataset_path,
                                  EXPORTED_FOLDER,
                                  SER_file[:-4] + 'train' + SER_file[-4:])

    test_filepath = os.path.join(dataset_path,
                                 EXPORTED_FOLDER,
                                 SER_file[:-4] + 'test' + SER_file[-4:])

    val_filepath = os.path.join(dataset_path,
                                EXPORTED_FOLDER,
                                SER_file[:-4] + 'val' + SER_file[-4:])

    # make sure files exist
    assert(os.path.exists(train_filepath))
    assert(os.path.exists(test_filepath))
    assert(os.path.exists(val_filepath))

    # load pairs from files
    # and keep only n_emotions
    # train
    train_pairs = import_csv(filepath=train_filepath)
    X_train, y_train = zip(*[(os.path.join(dataset_path, "AudioWAV", x[0]),
                              x[3])
                             for x in train_pairs if x[3] < n_emotions])
    # test
    test_pairs = import_csv(filepath=test_filepath)
    X_test, y_test = zip(*[(os.path.join(dataset_path, "AudioWAV", x[0]),
                            x[3])
                           for x in test_pairs if x[3] < n_emotions])
    # validation
    val_pairs = import_csv(filepath=val_filepath)
    X_val, y_val = zip(*[(os.path.join(dataset_path, "AudioWAV", x[0]),
                          x[3])
                         for x in val_pairs if x[3] < n_emotions])

    # format in file-label
    return X_train, y_train, X_test, y_test, X_val, y_val


if __name__ == "__main__":
    # Parse and export files & pairs & train val test
    file_pairs = read_label_file(dataset_path="datasets/crema-d")
    split_SER_SV(pairs=file_pairs,
                 n_emotions=6,
                 dataset_path="datasets/crema-d",
                 SER_file='SER_files.csv',
                 SV_file='SV_files.csv',
                 override=True)
    # return train test val split with labels
    load_CREMAD_SER(n_emotions=4)
