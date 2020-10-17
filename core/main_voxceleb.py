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


def train_se(e, dataloader, model, loss_function, optimizer, *args, **kwargs):
    """Training function for speaker embedder."""
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
    return training_loss / len(dataloader.dataset), None


def validate_se(e, dataloader, model, loss_function, *args, **kwargs):
    """Validation function for speaker embedder."""
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
    return validation_loss / len(dataloader.dataset), None


# Create all folders
dirs = [config.CHECKPOINT_FOLDER]
list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))
deterministic_model(deterministic=False)

# Split dataset to arrays
train_speakers, test_speakers, validation_speakers = load_VoxCeleb(val_ratio=.05,
                                                                   validation=True)
# lengths = []
# list(map(lambda sp: lengths.extend(glob.glob(sp + '/*/*.wav')),
#          train_speakers+validation_speakers))

# max_seq_len = compute_sequence_length_distribution(X=lengths)

print("Total information: 97.97 % by lowering max_sequence_length from 1450 to 245")
max_seq_len = 245

# PyTorch Settings
BATCH_SIZE = 16  # len(X_train) // 20
print(f'Selected Batch Size: {BATCH_SIZE}')
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


# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on: {device}.\n')

# #speakers # utterances, maxseqlen, melspectr_dim
N, M, width, height = next(iter(train_loader)).size()
# Set model and move to device
model = CNNSpeechEmbedder(height=height,
                          width=width).to(device)
logging.basicConfig(filename=config.LOG_FILE, level=logging.INFO)
print(f'Model Parameters: {model.count_parameters()}')
print(model)

#############################################################################
# Training Pipeline
#############################################################################

# Regularization parameters
L2 = 0
# Loss and optimizer
loss_function = GE2ELoss(device=device)
optimizer = torch.optim.AdamW([
    {'params': model.parameters()},
    {'params': loss_function.parameters()}
], weight_decay=0.02)


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

exit()
timestamp = time.ctime()

modelname = os.path.join(
    VARIABLES_FOLDER, f'{best_model.__class__.__name__}_{_epochs}_{timestamp}.pt')
# Save model for later use
torch.save(best_model, modelname)
# ===== TEST =====
y_pred, y_true = test(best_model, test_loader, cnn=CNN_BOOLEAN)
# ===== RESULTS =====
results(model=best_model, optimizer=optimizer, loss_function=loss_function,
        train_loss=train_losses, valid_loss=valid_losses,
        train_accuracy=train_accuracy, valid_accuracy=valid_accuracy,
        y_pred=y_pred, y_true=y_true, epochs=_epochs,
        cv=validation_epochs, timestamp=timestamp)
