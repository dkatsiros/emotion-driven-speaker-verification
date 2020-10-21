"""Exploiting GE2E Loss in a try to reduce EER on speaker verification. Dataset: Voxceleb 1"""
# System imports
import os
import random
import time
import logging

# External imports
import torch
from torch.utils.data import DataLoader
import glob2 as glob
from tqdm import tqdm

# Relative imports
import config
from config import VARIABLES_FOLDER, RECOMPUTE, DETERMINISTIC
from lib.loss import GE2ELoss
from lib.sound_processing import compute_max_sequence_length, compute_sequence_length_distribution
from lib.training import train_and_validate, test, results, overfit
from lib.training import deterministic_model
from models.SpeechEmbedder import CNNSpeechEmbedder
from dataloading.voxceleb import Voxceleb1
from utils.load_dataset import load_VoxCeleb


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
    return training_loss / len(dataloader.dataset)


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
    return validation_loss / len(dataloader.dataset)


def test_se(model, dataloader, testing_epochs=10):
    """Implementing test for speaker verification. Metric average Equal Error Rate."""
    from lib.loss import get_centroids, get_cossim
    # obtain the model's device ID
    device = next(model.parameters()).device

    avg_EER = 0
    for e in range(testing_epochs):
        batch_avg_EER = 0
        for batch_id, batch in enumerate(tqdm(dataloader)):
            # utterances should be even number to split exactly in two
            assert config.SPEAKER_M % 2 == 0
            # move to device
            batch = batch.to(device)
            # Seperate batch in M/2 and M/2 utterances per speaker
            enrollment_batch, verification_batch = torch.split(
                batch, int(config.SPEAKER_M/2), dim=1)
            # Reshape to (N*M,seq_len,features)
            enrollment_batch = torch.reshape(enrollment_batch,
                                             (config.SPEAKER_N*config.SPEAKER_M//2,
                                              1,
                                              enrollment_batch.size(2),
                                              enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch,
                                               (config.SPEAKER_N*config.SPEAKER_M//2,
                                                1,
                                                verification_batch.size(2),
                                                verification_batch.size(3)))
            # Shuffle verification batch
            perm = random.sample(
                range(0, verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i
            verification_batch = verification_batch[perm]

            # Forward through the network
            enrollment_embeddings = model(enrollment_batch)
            verification_embeddings = model(verification_batch)

            # Unshuffle
            verification_embeddings = verification_embeddings[unperm]

            # Restore shape (N,M,seq_len,features)
            enrollment_embeddings = torch.reshape(enrollment_embeddings,
                                                  (config.SPEAKER_N,
                                                   config.SPEAKER_M//2,
                                                   enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings,
                                                    (config.SPEAKER_N,
                                                     config.SPEAKER_M//2,
                                                     verification_embeddings.size(1)))

            enrollment_centroids = get_centroids(enrollment_embeddings)

            sim_matrix = get_cossim(embeddings=verification_embeddings,
                                    centroids=enrollment_centroids)

            # calculating EER
            diff = 1
            EER = 0
            EER_thresh = 0
            EER_FAR = 0
            EER_FRR = 0

            for thres in [0.01*i+0.5 for i in range(50)]:
                sim_matrix_thresh = sim_matrix > thres

                FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i, :, i].float().sum() for i in range(int(config.SPEAKER_N))])
                       / (config.SPEAKER_N-1.0)/(float(config.SPEAKER_M/2))/config.SPEAKER_N)

                FRR = (sum([config.SPEAKER_M/2-sim_matrix_thresh[i, :, i].float().sum() for i in range(int(config.SPEAKER_N))])
                       / (float(config.SPEAKER_M/2))/config.SPEAKER_N)

                # Save threshold when FAR = FRR (=EER)
                if diff > abs(FAR-FRR):
                    diff = abs(FAR-FRR)
                    EER = (FAR+FRR)/2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            logging.info("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" %
                         (EER, EER_thresh, EER_FAR, EER_FRR))
        avg_EER += batch_avg_EER/(batch_id+1)
    avg_EER = avg_EER / testing_epochs
    print("\n EER across {0} epochs: {1:.4f}".format(testing_epochs, avg_EER))


def train_voxceleb():
    """Load datasets, init models and train on VoxCeleb dataset."""

    # Split dataset to arrays
    train_speakers, _, validation_speakers = load_VoxCeleb(val_ratio=.05,
                                                           validation=True)
    # lengths = []
    # list(map(lambda sp: lengths.extend(glob.glob(sp + '/*/*.wav')),
    #          train_speakers+validation_speakers))

    # max_seq_len = compute_sequence_length_distribution(X=lengths)

    print("Total information: 97.97 % by lowering max_sequence_length from 1450 to 245")
    max_seq_len = 245

    # PyTorch Settings
    print(f'Selected Batch Size(#speakers per batch): {config.SPEAKER_N}')
    fe_method = "MEL_SPECTROGRAM" if config.CNN_BOOLEAN is True else "MFCC"

    # Training dataloader
    train_dataset = Voxceleb1(X=train_speakers,
                              training=True,
                              fe_method=fe_method,
                              max_seq_len=max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=config.SPEAKER_N,
                              shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True)
    # Validation dataloader
    validation_dataset = Voxceleb1(X=validation_speakers,
                                   validation=True,
                                   fe_method=fe_method,
                                   max_seq_len=max_seq_len)
    validation_loader = DataLoader(validation_dataset, batch_size=config.SPEAKER_N,
                                   shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True)

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

    print(f'Running on: {device}.\n')
    # logging & parameters
    logging.basicConfig(filename=config.LOG_FILE, level=logging.INFO)
    print(f'Model Parameters: {model.count_parameters()}')
    print(model)

    # Loss and optimizer
    loss_function = GE2ELoss(device=device)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': loss_function.parameters()}
    ], weight_decay=0.02)

    #############################################################################
    # Training Pipeline
    #############################################################################

    [best_model, train_losses,
     valid_losses, train_accuracy,
     valid_accuracy, _epochs] = train_and_validate(model=model,
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


def test_voxceleb(max_seq_len=245):
    """Load datasets, init models and test on VoxCeleb dataset."""

    # Split dataset to arrays
    _, test_speakers, _ = load_VoxCeleb(validation=False)

    fe_method = "MEL_SPECTROGRAM" if config.CNN_BOOLEAN is True else "MFCC"

    # Test dataloader
    test_dataset = Voxceleb1(X=test_speakers,
                             test=True,
                             fe_method=fe_method,
                             max_seq_len=max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=config.SPEAKER_N,
                             shuffle=True, num_workers=config.NUM_WORKERS, drop_last=True)

    N, M, width, height = next(iter(test_loader)).size()

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
    print(f'Model Parameters: {model.count_parameters()}')
    print(model)
    # Loss and optimizer
    loss_function = GE2ELoss(device=device)

    # Evaluation mode -gradients update off
    model.eval()
    # ===== TEST =====
    test_se(model, test_loader, testing_epochs=50)


if __name__ == "__main__":

    # Set everthing up before starting training & testing
    setup_environment_and_folders()
    # Core
    if config.TRAINING is True:
        train_voxceleb()
    else:
        test_voxceleb()
