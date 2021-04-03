import os
import numpy as np
import csv
import copy

EXPORTED_FOLDER = "emotional_speaker_verification_exported"


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
            tuples.append(row)
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


def split_SER_SV(pairs=[],
                 dataset_path="datasets/crema-d",
                 SER_file='SER_files.csv',
                 SV_file='SV_files.csv',
                 override=False):
    """Load files with labels and slit to male-female to avoid
    trivial solutions"""

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
    filepath = os.path.join(dataset_path, EXPORTED_FOLDER, SER_file)
    export_csv(tuples=SER_samples, filepath=filepath, override=override)

    ##
    # Speaker Verification
    # try to export csv, if not already there
    filepath = os.path.join(dataset_path, EXPORTED_FOLDER, SV_file)
    export_csv(tuples=SV_samples, filepath=filepath, override=override)


if __name__ == "__main__":
    file_pairs = read_label_file(dataset_path="datasets/crema-d")
    split_SER_SV(pairs=file_pairs,
                 dataset_path="datasets/crema-d",
                 SER_file='SER_files.csv',
                 SV_file='SV_files.csv',
                 override=True)
