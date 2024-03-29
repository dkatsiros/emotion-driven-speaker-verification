from lib.training import overfit_batch
import os
import random
import time
import logging
from copy import deepcopy

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np

import config
from config import VARIABLES_FOLDER, RECOMPUTE
from lib.training import train_and_validate, test, results, deterministic_model
from utils.load_dataset import load_IEMOCAP
from dataloading.iemocap import IemocapDataset
from models.lstm import LSTM
from models.EmotionalSpeechEmbedder import CNNEmotionalSpeechEmbedder
from plotting.class_stats import dataloader_stats, samples_lengths

DATASET = "IEMOCAP"

# Deterministic ?
deterministic_model(config.DETERMINISTIC)


N_CLASSES = 4  # 4 #6 #9

# Split dataset to arrays
[X_train, y_train,
 X_test, y_test,
 X_eval, y_eval] = load_IEMOCAP(n_classes=N_CLASSES, test_val=[0.1, 0.05])

# PyTorch
BATCH_SIZE = config.BATCH_SIZE  # len(X_train) // 20
print(f'Selected Batch Size: {BATCH_SIZE}')
EPOCHS = config.EPOCHS


CNN_BOOLEAN = True
max_seq_len = 245
fe_method = feature_extraction_method = "MEL_SPECTROGRAM" if CNN_BOOLEAN is True else "MFCC"

# Load sets using dataset class
train_set = IemocapDataset(X_train, y_train,
                           oversampling=True,
                           feature_extraction_method=fe_method,
                           fixed_length=True,
                           max_seq_len=max_seq_len)

test_set = IemocapDataset(X_test, y_test,
                          oversampling=False,
                          feature_extraction_method=fe_method,
                          fixed_length=True,
                          max_seq_len=max_seq_len)

eval_set = IemocapDataset(X_eval, y_eval,
                          oversampling=False,
                          feature_extraction_method=fe_method,
                          fixed_length=True,
                          max_seq_len=max_seq_len)


# PyTorch DataLoader

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          num_workers=4, drop_last=True, shuffle=True)

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                         num_workers=4, drop_last=True, shuffle=True)

valid_loader = DataLoader(eval_set, batch_size=BATCH_SIZE,
                          num_workers=4, drop_last=True, shuffle=True)

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on: {device}.\n')

# Define CNN internal dimensions based on predifined batch
# batch_size, height, width = next(iter(valid_loader)).size()
height = 128
width = max_seq_len
# initalize model
model = CNNEmotionalSpeechEmbedder(height=height,
                                   width=width,
                                   out_fe=N_CLASSES)
print(f'Model Parameters: {model.count_parameters()}')

# move model weights to device
# logging & parameters
model.to(device)
print(model)

#############################################################################
# Training Pipeline
#############################################################################
# Loss weights
# weights = torch.tensor([1.25,1,1,1]).type('torch.FloatTensor').to(device)#,1,1,1,1,1] # sad =2
# Loss and optimizer
loss_function = torch.nn.CrossEntropyLoss()  # weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.02)

# overfit_batch(model, train_loader, loss_function,
#               optimizer, 100, cnn=config.CNN_BOOLEAN)
# exit()
logging.basicConfig(filename=config.LOG_FILE, level=logging.INFO)
best_model, _epochs = train_and_validate(model=model,
                                         train_loader=train_loader,
                                         valid_loader=valid_loader,
                                         loss_function=loss_function,
                                         optimizer=optimizer,
                                         epochs=config.EPOCHS,
                                         cnn=config.CNN_BOOLEAN,
                                         valid_freq=config.VALID_FREQ,
                                         early_stopping=True)

timestamp = time.ctime()

modelname = os.path.join(
    VARIABLES_FOLDER, f'{best_model.__class__.__name__}_{_epochs}_{timestamp}.pt')
# Save model for later use
torch.save(best_model.state_dict(), modelname)

# ===== TEST =====
y_pred, y_true = test(best_model, test_loader, cnn=CNN_BOOLEAN)
# ===== RESULTS =====
results(model=best_model, optimizer=optimizer, loss_function=loss_function,
        y_pred=y_pred, y_true=y_true, epochs=_epochs,
        timestamp=timestamp, dataset=DATASET, n_classes=N_CLASSES)
