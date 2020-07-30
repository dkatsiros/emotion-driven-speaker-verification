import numpy as np
import time
from copy import deepcopy
import torch
from torch import optim
from utils.load_dataset import load_IEMOCAP
from dataloading.iemocap import IemocapDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F

from models.lstm import LSTM
from models.cnn import CNN
from models.cnn2 import CNN2
from models.cnn3 import CNN3
from lib.training import train_and_validate, test, results, overfit
# progress, fit, print_results
from config import VARIABLES_FOLDER, RECOMPUTE, DETERMINISTIC
import os
import random
from plotting.class_stats import dataloader_stats, samples_lengths

DATASET = "IEMOCAP"

# Deterministic ?
if DETERMINISTIC is True:
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


N_CLASSES = 4

# Split dataset to arrays
[X_train, y_train,
 X_test, y_test,
 X_eval, y_eval] = load_IEMOCAP(n_classes=N_CLASSES)

# PyTorch
BATCH_SIZE = 16  # len(X_train) // 20
print(f'Selected Batch Size: {BATCH_SIZE}')
EPOCHS = 500

CNN_BOOLEAN = True

# Load sets using dataset class
train_set = IemocapDataset(X_train, y_train, oversampling=True,
                           feature_extraction_method="MEL_SPECTROGRAM" if CNN_BOOLEAN is True else "MFCC")

test_set = IemocapDataset(
    X_test, y_test, feature_extraction_method="MEL_SPECTROGRAM" if CNN_BOOLEAN is True else "MFCC")

eval_set = IemocapDataset(
    X_eval, y_eval, feature_extraction_method="MEL_SPECTROGRAM" if CNN_BOOLEAN is True else "MFCC")


# PyTorch DataLoader
try:
    TRAIN_LOADER = os.path.join(VARIABLES_FOLDER, 'train_loader.pt')
    if RECOMPUTE is True:
        raise NameError('Forced recomputing values.')
    train_loader = torch.load(TRAIN_LOADER)
except:
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, num_workers=4, drop_last=True, shuffle=True)
    torch.save(train_loader, TRAIN_LOADER)
dataloader_stats(
    train_loader, filename='train_loader_statistics.png', dataset=DATASET)

try:
    TEST_LOADER = os.path.join(VARIABLES_FOLDER, 'test_loader.pt')
    if RECOMPUTE is True:
        raise NameError('Forced recomputing values.')
    test_loader = torch.load(TEST_LOADER)
except:
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, num_workers=4, drop_last=True, shuffle=True)
    torch.save(test_loader, TEST_LOADER)
    dataloader_stats(
        test_loader, filename='test_loader_statistics.png', dataset=DATASET)

try:
    VALID_LOADER = os.path.join(VARIABLES_FOLDER, 'valid_loader.pt')
    if RECOMPUTE is True:
        raise NameError('Forced recomputing values.')
    valid_loader = torch.load(VALID_LOADER)
except:
    valid_loader = DataLoader(
        eval_set, batch_size=BATCH_SIZE, num_workers=4, drop_last=True, shuffle=True)
    torch.save(valid_loader, VALID_LOADER)
    dataloader_stats(
        valid_loader, filename='valid_loader_statistics.png', dataset=DATASET)

# Print sequence length diagram for samples
# samples_lengths(dataloaders=[train_loader, valid_loader])

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on: {device}.\n')

# Create a model
# model = LSTM(input_size=39, hidden_size=6, output_size=7, num_layers=3,
#              bidirectional=True, dropout=0.2)

model = CNN3(output_dim=N_CLASSES)
print(f'Model Parameters: {model.count_parameters(model)}')

# move model weights to device
model.to(device)
print(model)


#############################################################################
# Training Pipeline
#############################################################################

print(next(iter(train_loader)))
# Regularization parameters
learning_rate = 1e-5
L2 = 0
# Loss and optimizer
loss_function = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adadelta(
#     model.parameters(), rho=0.9, weight_decay=L2)  # eps = 1e-06
# optimizer = torch.optim.SGD(params=model.parameters(
# ), lr=learning_rate, momentum=0.9, weight_decay=L2)
# optimizer = torch.optim.Adam(model.parameters(), weight_decay=L2)
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.02)
CROSS_VALIDATION_EPOCHS = 5

# Test overfit
# model, all_train_loss, epoch = overfit(model,
#                                       train_loader,
#                                       loss_function,
#                                       optimizer,
#                                       epochs=EPOCHS,
#                                       cnn=CNN_BOOLEAN)
# exit()
best_model, train_losses, valid_losses, train_accuracy, valid_accuracy, _epochs = train_and_validate(model=model,
                                                                                                     train_loader=train_loader,
                                                                                                     valid_loader=valid_loader,
                                                                                                     loss_function=loss_function,
                                                                                                     optimizer=optimizer,
                                                                                                     epochs=EPOCHS,
                                                                                                     cnn=CNN_BOOLEAN,
                                                                                                     cross_validation_epochs=5,
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
        train_loss=train_losses, valid_loss=valid_losses,
        train_accuracy=train_accuracy, valid_accuracy=valid_accuracy,
        y_pred=y_pred, y_true=y_true, epochs=_epochs,
        cv=CROSS_VALIDATION_EPOCHS, timestamp=timestamp, dataset=DATASET)