"""Speaker Verification for emotional content on voxceleb """
# System imports
import os
import random
import time
import logging

# External imports
import torch
from torch.utils.data import DataLoader
import numpy as np
import glob2 as glob
from tqdm import tqdm

# Relative imports
import config
from config import VARIABLES_FOLDER, RECOMPUTE, DETERMINISTIC
from lib.metrics import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf
from lib.loss import GE2ELoss
from lib.model_editing import drop_layers, print_require_grad_parameter
from lib.sound_processing import compute_max_sequence_length, compute_sequence_length_distribution
from lib.training import train_and_validate, test, results, overfit_batch
from lib.training import deterministic_model
from models.SpeechEmbedder import CNNSpeechEmbedder
from dataloading.ravdess import RAVDESS_Evaluation_PreComputedMelSpectr


def setup_environment_and_folders():
    # Create all folders
    dirs = [config.CHECKPOINT_FOLDER]
    list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))
    deterministic_model(deterministic=config.DETERMINISTIC)


def test_se(model, dataloader, testing_epochs=10):
    """Implementing test for speaker verification. Metric average Equal Error Rate."""
    # obtain the model's device ID
    device = next(model.parameters()).device

    avg_EER = []
    avg_mindcf = []
    for e in range(testing_epochs):
        epoch_avg_EER = 0
        batch_mindcf = 0
        # dataloader: (batch_size,2,max_seq_len,input_fe)
        for batch_id, (batch, labels) in enumerate(tqdm(dataloader)):
            # move to device
            batch = batch.to(device, dtype=torch.float)
            labels = labels.to(device)
            # prepare for forward pass
            batch = torch.reshape(batch,
                                  (batch.size(0)*2, 1, batch.size(2), batch.size(3)))
            # pass the batch through the model
            embeddings = model(batch)
            # reshape to the original (batch_size,2,fe)
            embeddings = torch.reshape(embeddings,
                                       (batch.size(0)//2, 2, -1))
            # split to e1 & e2 for each test pair
            enrollment, verification = torch.split(embeddings,
                                                   1,  # split_size
                                                   dim=1)  # final 2* (batch_size,1,out_dim)
            enrollment = torch.squeeze(enrollment)  # reduce dim=1
            verification = torch.squeeze(verification)  # reduce dim=1
            # compute the cosine similarity (batch_size,1)
            cossim = torch.nn.functional.cosine_similarity(enrollment,
                                                           verification)

            # batch and labels back to cpu
            batch.cpu()
            labels.cpu()

            # Fast EER computation
            tunedThreshold, batch_EER, fpr, fnr = tuneThresholdfromScore(cossim.cpu().detach(),
                                                                         labels.cpu(),
                                                                         [1, 0.1])
            fnrs, fprs, thresholds = ComputeErrorRates(cossim.cpu().detach(),
                                                       labels.cpu())
            # eer=(far + frr)/2
            epoch_avg_EER = (batch_id * epoch_avg_EER +
                             batch_EER)/(batch_id+1)

            p_target = 0.01
            c_miss = 1
            c_fa = 1

            mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds,
                                      p_target, c_miss, c_fa)
            batch_mindcf = (batch_id * batch_mindcf + mindcf)/(batch_id+1)

            logging.info(f"\navg_EER (epoch:{ e+1 }): {epoch_avg_EER:.2f}")
            logging.info(f"\nmin DCF (epoch: {e+1}): {mindcf:.2f}")

        # Get mean of #testing_epochs EER
        avg_EER.append(epoch_avg_EER)
        avg_mindcf.append(batch_mindcf)

    print("\n avg_EER across {0} epochs: {1:.2f} \pm {2:.2f}".format(
        testing_epochs, np.mean(avg_EER), np.std(avg_EER)))
    print("\n min_dcf across {0} epochs: {1:.2f} \pm {2:.2f}".format(
        testing_epochs, np.mean(avg_mindcf), np.std(avg_mindcf)))
    return (round(np.mean(avg_EER), 2), round(np.std(avg_EER), 2),
            round(np.mean(avg_mindcf), 4), round(np.std(avg_mindcf), 4))


def test_ravdess(model=config.MODEL_TO_RESTORE,
                 verification_file=config.TEST_FILE_PATH,
                 max_seq_len=245):
    """Load datasets, init models and test on VoxCeleb dataset."""

    fe_method = "MEL_SPECTROGRAM" if config.CNN_BOOLEAN is True else "MFCC"
    dataset_func = RAVDESS_Evaluation_PreComputedMelSpectr
    # Test dataloader
    test_dataset = dataset_func(test_file_path=verification_file,
                                fe_method=fe_method,
                                max_seq_len=max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=True, num_workers=config.NUM_WORKERS, drop_last=False)

    batch_, label_ = next(iter(test_loader))
    (N, M, width, height), label_size = batch_.size(), label_.size()
    # if your computer has a CUDA compatible gpu use it, otherwise use the cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Selected Batch Size: {config.BATCH_SIZE}')

    # Restore trained model
    try:
        model = torch.load(model, map_location=device)
    except Exception as model_can_not_be_restored:
        raise FileNotFoundError from model_can_not_be_restored

    # logging & parameters
    logging.basicConfig(filename=config.LOG_FILE_TEST, level=logging.INFO)
    print(f'Running on: {device}.\n')
    # print(f'Model Parameters: {model.count_parameters()}')
    print(model)

    # Evaluation mode -gradients update off
    model.eval()
    # ===== TEST =====
    mean_eer, std_eer, mean_dcf, std_dcf = test_se(
        model, test_loader, testing_epochs=50)
    return mean_eer, std_eer, mean_dcf, std_dcf


def export_visual_representation_2D(ofile=None):
    if os.path.exists(ofile):
        raise FileExistsError()
    fe_method = "MEL_SPECTROGRAM" if config.CNN_BOOLEAN is True else "MFCC"
    dataset_func = RAVDESS_Evaluation_PreComputedMelSpectr
    # Test dataloader
    test_dataset = dataset_func(test_file_path=config.TEST_FILE_PATH,
                                fe_method=fe_method,
                                max_seq_len=245,
                                details=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=True, num_workers=config.NUM_WORKERS, drop_last=False)
    # Restore trained model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(config.MODEL_TO_RESTORE, map_location=device)
    except Exception as model_can_not_be_restored:
        raise FileNotFoundError from model_can_not_be_restored

    model.eval()

    speaker_space = np.empty((24, 4, 8, 256))

    for batch_id, (batch, labels, speakers, emotions, statement, repetition) in enumerate(tqdm(test_loader)):
        # move to device
        batch = batch.to(device, dtype=torch.float)
        labels = labels.to(device)
        # prepare for forward pass
        batch = torch.reshape(batch,
                              (batch.size(0)*2, 1, batch.size(2), batch.size(3)))
        # pass the batch through the model
        embeddings = model(batch)
        # reshape to the original (batch_size,2,fe)
        embeddings = torch.reshape(embeddings,
                                   (batch.size(0)//2, 2, -1))
        # split to e1 & e2 for each test pair
        enrollment, verification = torch.split(embeddings,
                                               1,  # split_size
                                               dim=1)  # final 2* (batch_size,1,out_dim)
        enrollment = torch.squeeze(enrollment)  # reduce dim=1
        verification = torch.squeeze(verification)  # reduce dim=1

        speaker_space[speakers[0],
                      2*statement[0]+repetition[0],
                      emotions[0],
                      :] = enrollment.cpu().detach().numpy()
        speaker_space[speakers[1],
                      2*statement[1]+repetition[1],
                      emotions[1],
                      :] = verification.cpu().detach().numpy()

        # batch and labels back to cpu
        batch.cpu()
        labels.cpu()
    np.save(ofile, speaker_space)


def visualize(ifile, ofile):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    speaker_space = np.load(ifile)  # (24,4,8,256)

    markers = ["_", "*", "^", "v", "x", "o", "2", "P"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#2f77b4', '#3f7f0e', '#4ca02c', '#562728', '#6467bd',
              '#2c564b', '#3377c2', '#4f7f7f', '#5cbd22', '#67becf',
              '#1f77b4', '#0f7f0e', '#9ca02c', '#862728', '#7467bd']
    pca = PCA(n_components=2)

    features = speaker_space.reshape((24*4*8, 256))

    features = pca.fit_transform(features)

    print(features.shape)

    features = features.reshape((24, 4, 8, 2))

    fig = plt.figure(figsize=(12, 12))
    for speaker in range(6):
        for emotion in range(3, 5):
            for statement in range(4):
                plt.scatter(features[speaker, statement, emotion, 0], features[speaker, statement, emotion, 1],
                            marker=markers[emotion],
                            s=100,
                            c=colors[speaker],
                            label=f"spkr_{speaker}")
    plt.savefig(ofile)
    # pca.fit


def test_ravdess_and_export(model=None, verification_file=None, ofile=None):
    """
Automate testing models with custom verification files and exporting
the results to a file
"""
    if model is None or verification_file is None or ofile is None:
        raise AssertionError()

    mean_eer, std_eer, mean_dcf, std_dcf = test_ravdess(model=model,
                                                        verification_file=verification_file)

    with open(ofile, mode="a") as file:
        file.write(
            f"{model}\t{verification_file}\t{mean_eer}\t{std_eer}\t{mean_dcf}\t{std_dcf}\n")


def export_latex(ifile=None, ofile=None):
    if ifile is None:
        raise AssertionError()
    with open(ifile, mode="r") as file:
        data = file.readlines()
    exp_results = []
    for line in data:
        model, verification_file, mean_eer, std_eer, mean_dcf, std_dcf = line.replace(
            "\n", "").split("\t")
        exp_results.append(
            (model, verification_file, mean_eer, std_eer, mean_dcf, std_dcf))
    if ofile is None:
        ofile = ifile[:-3] + '.tex'
    HEADER = """\\begin{tabular}{ |p{0.8cm}|p{6cm}||p{2cm}|p{2cm}| }
 \hline
 \multicolumn{4}{|c|}{Neutral enrollment and \\textit{weak or strong} emotionally injected verification utterance} \\\\
 \hline
 Exp & emotion & weak & strong \\\\
 \hline
"""
    emotion_names = ["neutral", "calm", "happy", "sad",
                     "angry", "fearful", "disgust", "surprised"]

    with open(ofile, mode="w") as file:
        file.write(HEADER)
        for data, data_intense in zip(exp_results[::2], exp_results[1::2]):
            # Weak emotion
            model, verification_file, mean_eer, std_eer, mean_dcf, std_dcf = data
            exp_details = verification_file[-9:][:-4]
            exp_num, intensity, emotion = exp_details.split('.')
            result1 = "\SI{"+mean_eer + "\pm" + std_eer + "}"
            # Strong emotion
            model, verification_file, mean_eer, std_eer, mean_dcf, std_dcf = data_intense
            exp_details = verification_file[-9:][:-4]
            exp_num, intensity, emotion = exp_details.split('.')
            result2 = "\SI{"+mean_eer + "\pm" + std_eer + "}"
            # write file
            file.write(
                f""" {exp_num}.{emotion} & {emotion_names[int(emotion)]}  & ${result1}$ & ${result2}$\\\\\n""")

        file.write(""" \hline
 \end{tabular} \\break\\break\\break
""")


def exp4_export_latex(model=None, ofile=None):
    ifile_ignorance = f"results/exp4.1.{model[:-3]}.txt"
    ifile_knowledge = f"results/exp4.2.{model[:-3]}.txt"
    # ifile
    if not os.path.exists(ifile_ignorance):
        raise FileNotFoundError(ifile_ignorance)
    if not os.path.exists(ifile_knowledge):
        raise FileNotFoundError()

    # store results
    exp_results_ignorance = []
    with open(ifile_ignorance, mode="r") as file:
        data = file.readlines()
    # read exported results - ignorance
    for line in data:
        _, verification_file, mean_eer, std_eer, mean_dcf, std_dcf = line.replace(
            "\n", "").split("\t")
        exp_results_ignorance.append(
            (model, verification_file, mean_eer, std_eer, mean_dcf, std_dcf))

    # store results
    exp_results_knwoledge = []
    with open(ifile_knowledge, mode="r") as file:
        data = file.readlines()
    # read exported results - knowledge
    for line in data:
        _, verification_file, mean_eer, std_eer, mean_dcf, std_dcf = line.replace(
            "\n", "").split("\t")
        exp_results_knwoledge.append(
            (model, verification_file, mean_eer, std_eer, mean_dcf, std_dcf))

    if ofile is None:
        ofile = f"results/results_exp4.{model[:-3]}.tex"

    HEADER = """\\begin{tabular}{ |p{0.8cm}|p{2cm}||p{2cm}|p{2cm}| }
 \hline
 \multicolumn{4}{|c|}{Emotion ignorance vs emotion knowledge for model } \\\\
 \hline
 Exp & emotion & ignorance & knowledge \\\\
 \hline
"""
    emotion_names = ["neutral", "calm", "happy", "sad",
                     "angry", "fearful", "disgust", "surprised"]

    with open(ofile, mode="w") as file:
        file.write(HEADER)
        for idx, (ignorance, knowledge) in enumerate(zip(exp_results_ignorance, exp_results_knwoledge), 1):
            # ignorance
            model, verification_file, mean_eer, std_eer, mean_dcf, std_dcf = ignorance
            # exp_details = verification_file[-9:][:-4]
            # exp_num, intensity, emotion = exp_details.split('.')
            result1 = "\SI{"+mean_eer + "\pm" + std_eer + "}"
            # Strong emotion
            model, verification_file, mean_eer, std_eer, mean_dcf, std_dcf = knowledge
            # exp_details = verification_file[-9:][:-4]
            # exp_num, intensity, emotion = exp_details.split('.')
            result2 = "\SI{"+mean_eer + "\pm" + std_eer + "}"
            # write file
            file.write(
                f""" 4.{idx} & {emotion_names[int(idx)]}  & ${result1}$ & ${result2}$\\\\\n""")

        file.write(""" \hline
 \end{tabular} \\break\\break\\break
""")


if __name__ == "__main__":

    # Set everthing up before starting training & testing
    setup_environment_and_folders()
    # Core
    # mean_eer, std_eer, mean_dcf, std_dcf = test_ravdess()

    # export_visual_representation_2D(
    #     ofile="datasets/ravdess/representation_exp2.1.npy")

    # visualize("datasets/ravdess/representation_exp1.1.npy",
    #           "datasets/ravdess/latent_space_exp1.1.png")
    # model = "checkpoints/emot_frozenconv_voxceleb_lr=.5e-1.pt"
    # ofile = f"results/exp3.{model.split('/')[-1][:-3]}.txt"
    # for emotion in [1, 2, 3, 4, 5, 6, 7]:
    #     for intensity in [1, 2]:
    #         verification_file = f"datasets/ravdess/veri_files/veri_test_exp3.{intensity}.{emotion}.txt"
    #         test_ravdess_and_export(model=model,
    #                                 verification_file=verification_file,
    #                                 ofile=ofile)

    # export_latex(ifile=ofile)
    exp4_export_latex(model="fusion_model_lr=1e-3.pt")
#
