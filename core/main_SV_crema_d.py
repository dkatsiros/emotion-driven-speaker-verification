"""Exploiting GE2E Loss in a try to reduce EER on speaker verification. Dataset: Voxceleb 1"""
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
from dataloading.crema_d import CremaDPrecomputed, CremaD_Evaluation_PreComputedMelSpectr
from utils.crema_d import load_CREMAD_SV


def setup_environment_and_folders():
    # Create all folders
    dirs = [config.CHECKPOINT_FOLDER]
    list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))
    deterministic_model(deterministic=config.DETERMINISTIC)


def train_se(e, dataloader, model, loss_function, optimizer, *args, **kwargs):
    """Training function for speaker embedder."""
    # obtain the model's device ID
    device = next(model.parameters()).device

    model.train()
    training_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Training Epoch {e}"):
        batch = batch.to(device)

        batch = torch.reshape(batch,
                              (config.SPEAKER_N*config.SPEAKER_M, 1, batch.size(2), batch.size(3)))
        perm = random.sample(range(0, config.SPEAKER_N*config.SPEAKER_M),
                             config.SPEAKER_N*config.SPEAKER_M)
        unperm = list(perm)
        for i, j in enumerate(perm):
            unperm[j] = i
        batch = batch[perm]
        # gradient accumulates
        optimizer.zero_grad()

        embeddings = model(batch)
        embeddings = embeddings[unperm]
        embeddings = torch.reshape(embeddings,
                                   (config.SPEAKER_N, config.SPEAKER_M, embeddings.size(1)))

        # get loss, call backward, step optimizer
        # wants (Speaker, Utterances, embedding)
        loss = loss_function(embeddings)
        training_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        torch.nn.utils.clip_grad_norm_(loss_function.parameters(), 1.0)
        optimizer.step()
    return training_loss / (len(dataloader.dataset)/dataloader.batch_size)


def validate_se(e, dataloader, model, loss_function, *args, **kwargs):
    """Validation function for speaker embedder."""
    # obtain the model's device ID
    device = next(model.parameters()).device

    model.eval()
    validation_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Validation Epoch {e}"):
        batch = batch.to(device)

        batch = torch.reshape(batch,
                              (config.SPEAKER_N*config.SPEAKER_M, 1, batch.size(2), batch.size(3)))
        perm = random.sample(range(0, config.SPEAKER_N*config.SPEAKER_M),
                             config.SPEAKER_N*config.SPEAKER_M)
        unperm = list(perm)
        for i, j in enumerate(perm):
            unperm[j] = i
        batch = batch[perm]
        embeddings = model(batch)
        embeddings = embeddings[unperm]
        embeddings = torch.reshape(embeddings,
                                   (config.SPEAKER_N, config.SPEAKER_M, embeddings.size(1)))

        # get loss, call backward, step optimizer
        # wants (Speaker, Utterances, embedding)
        loss = loss_function(embeddings)
        validation_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        torch.nn.utils.clip_grad_norm_(loss_function.parameters(), 1.0)
    return validation_loss / (len(dataloader.dataset)/dataloader.batch_size)


def test_se(model, dataloader, testing_epochs=10):
    """Implementing test for speaker verification. Metric average Equal Error Rate."""
    # obtain the model's device ID
    device = next(model.parameters()).device

    avg_EER = []
    avg_mindcf = []
    all_cossim = []
    all_labels = []
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

            # total computation
            all_cossim.extend(cossim.cpu().detach())
            all_labels.extend(labels.cpu())
            # tunedThreshold, batch_EER, fpr, fnr = tuneThresholdfromScore(cossim.cpu().detach(),
            #                                                              labels.cpu(),
            #                                                              [1, 0.1])
            # fnrs, fprs, thresholds = ComputeErrorRates(cossim.cpu().detach(),
            #                                            labels.cpu())
            # eer=(far + frr)/2
            # epoch_avg_EER = (batch_id * epoch_avg_EER +
            #                  batch_EER)/(batch_id+1)

            # p_target = 0.01
            # c_miss = 1
            # c_fa = 1

            # mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds,
            #                           p_target, c_miss, c_fa)
            # batch_mindcf = (batch_id * batch_mindcf + mindcf)/(batch_id+1)

            # logging.info(f"\navg_EER (epoch:{ e+1 }): {epoch_avg_EER:.2f}")
            # logging.info(f"\nmin DCF (epoch: {e+1}): {mindcf:.2f}")

        # Get mean of #testing_epochs EER
        # avg_EER.append(epoch_avg_EER)
        # avg_mindcf.append(batch_mindcf)
        tunedThreshold, EER, fpr, fnr = tuneThresholdfromScore(all_cossim,
                                                               all_labels,
                                                               [1, 0.1])

        avg_EER.append(EER)
        print(f"\navg_EER (epoch:{ e+1 }): {EER:.2f}")
        logging.info(f"\navg_EER (epoch:{ e+1 }): {epoch_avg_EER:.2f}")
    # print("\n avg_EER across {0} epochs: {1:.4f} +- {2:.4f}".format(
    #     testing_epochs, np.mean(avg_EER), np.std(avg_EER)))
    # print("\n min_dcf across {0} epochs: {1:.4f} +- {2:.4f}".format(
    #     testing_epochs, np.mean(avg_mindcf), np.std(avg_mindcf)))


def train_SV_crema_d():
    """Load datasets, init models and train on Crema-D dataset."""

    # Split dataset to arrays
    train_speakers, validation_speakers = load_CREMAD_SV()
    # lengths = []
    # list(map(lambda sp: lengths.extend(glob.glob(sp + '/*/*.wav')),
    #          train_speakers+validation_speakers))

    # max_seq_len = compute_sequence_length_distribution(X=lengths)

    max_seq_len = 46
    print(f"Max sequence length = {max_seq_len}")
    # PyTorch Settings
    print(f'Selected Batch Size(#speakers per batch): {config.SPEAKER_N}')
    fe_method = "MEL_SPECTROGRAM" if config.CNN_BOOLEAN is True else "MFCC"

    dataset_func = CremaDPrecomputed
    # Training dataloader
    train_dataset = dataset_func(X=train_speakers,
                                 training=True,
                                 fe_method=fe_method,
                                 max_seq_len=max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=config.SPEAKER_N,
                              shuffle=False, num_workers=config.NUM_WORKERS, drop_last=True)
    # Validation dataloader
    validation_dataset = dataset_func(X=validation_speakers,
                                      validation=True,
                                      fe_method=fe_method,
                                      max_seq_len=max_seq_len)
    validation_loader = DataLoader(validation_dataset, batch_size=config.SPEAKER_N,
                                   shuffle=False, num_workers=config.NUM_WORKERS, drop_last=True)

    # #speakers # utterances, maxseqlen, melspectr_dim
    N, M, width, height = next(iter(train_loader)).size()

    # if your computer has a CUDA compatible gpu use it, otherwise use the cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create new model or restore a semi-trained
    if config.RESTORE_FROM_PREVIOUS_MODEL is True:
        try:
            model = torch.load(config.MODEL_TO_RESTORE, map_location=device)
        except Exception as model_can_not_be_restored:
            raise FileNotFoundError from model_can_not_be_restored
    else:
        model = CNNSpeechEmbedder(height=height,
                                  width=width).to(device)

    # Remove final linear layer
    # making output 256 dimension
    model = drop_layers(model, 1)
    print_require_grad_parameter(model)

    print(f'Running on: {device}.\n')
    # logging & parameters
    logging.basicConfig(filename=config.LOG_FILE, level=logging.INFO)
    # print(f'Model Parameters: {model.count_parameters()}')
    print(model)

    # Loss and optimizer
    loss_function = GE2ELoss(device=device)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': loss_function.parameters()}
    ], weight_decay=0.02, lr=(config.LEARNING_RATE if config.LEARNING_RATE is not None else 1e-3))

    #############################################################################
    # Training Pipeline
    #############################################################################

    [best_model,  _epochs] = train_and_validate(model=model,
                                                train_loader=train_loader,
                                                valid_loader=validation_loader,
                                                loss_function=loss_function,
                                                optimizer=optimizer,
                                                epochs=config.EPOCHS,
                                                cnn=config.CNN_BOOLEAN,
                                                valid_freq=config.VALID_FREQ,
                                                early_stopping=True,
                                                train_func=train_se,
                                                validate_func=validate_se)


def test_SV_crema_d(max_seq_len=46):
    """Load datasets, init models and test on VoxCeleb dataset."""

    fe_method = "MEL_SPECTROGRAM" if config.CNN_BOOLEAN is True else "MFCC"
    dataset_func = CremaD_Evaluation_PreComputedMelSpectr
    # Test dataloader
    test_dataset = dataset_func(test_file_path=config.TEST_FILE_PATH,
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
        model = torch.load(config.MODEL_TO_RESTORE, map_location=device)
    except Exception as model_can_not_be_restored:
        raise FileNotFoundError from model_can_not_be_restored

    # logging & parameters
    logging.basicConfig(filename=config.LOG_FILE_TEST, level=logging.INFO)
    print(f'Running on: {device}.\n')
    # print(f'Model Parameters: {model.count_parameters()}')
    print(model)
    # Loss and optimizer
    loss_function = GE2ELoss(device=device)

    # Evaluation mode -gradients update off
    model.eval()
    # ===== TEST =====
    test_se(model, test_loader, testing_epochs=1)


if __name__ == "__main__":

    # Set everthing up before starting training & testing
    setup_environment_and_folders()
    # Core
    if config.TRAINING is True:
        train_SV_crema_d()
    else:
        test_SV_crema_d()
