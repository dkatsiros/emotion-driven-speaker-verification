import glob2 as glob
from tqdm import tqdm
from lib.sound_processing import load_wav, get_melspectrogram

paths = ['datasets/voxceleb1/train/wav',
         'datasets/voxceleb1/test/wav']

corrupted = []
for path in paths:
    for wav in tqdm(glob.glob(path + '/*/*/*.wav'), desc=f"{path}"):
        try:
            get_melspectrogram(load_wav(wav))
        except:
            print(wav)
            corrupted.append(wav)

with open("corrupted_wavs.txt", mode="a") as file:
    for c in corrupted:
        file.write(f"{c}\n")
