"""Given a path returns the distribution of max sequence length for all the wav files."""
import os
import argparse
import numpy as np
import glob2 as glob
from lib.sound_processing import load_wav, get_melspectrogram


def get_max_seq_len(paths=[]):
    iterators_for_paths = []
    lengths = []
    for path in paths:
        path_reg = os.path.join(path, "**/*.wav")
        wavfiles = glob.iglob(path_reg, recursive=True)
        iterators_for_paths.append(wavfiles)
    for iterator in iterators_for_paths:
        # Calculate and print max sequence number
        lengths.extend([np.shape(get_melspectrogram(load_wav(f)))[0]
                        for f in iterator])

    print(f"Max sequence lenght computed: {np.max(lengths)}")
    print(f"Mean: {np.mean(lengths)}")
    print(f"Mean: {np.std(lengths)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Get paths..")
    parser.add_argument('--path', metavar='p', type=str, nargs="+", default=[])
    args = parser.parse_args()
    paths = args.path
    get_max_seq_len(paths=paths)
