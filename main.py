import torch
from torch import optim
from utils.load_dataset import load_Emodb
from dataloading import EmodbDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F

from models.lstm import LSTM
from training import train, validate, progress, fit, print_results
from config import VARIABLES_FOLDER, RECOMPUTE
import os
import joblib

# Split dataset to arrays
X_train, y_train, X_test, y_test, X_eval, y_eval = load_Emodb()

# PyTorch
BATCH_SIZE = len(X_train) // 20
EPOCHS = 50

# Load sets using dataset class
train_set = EmodbDataset(X_train, y_train, oversampling=True)
test_set = EmodbDataset(X_test, y_test)
eval_set = EmodbDataset(X_eval, y_eval)


# PyTorch DataLoader
try:
    TRAIN_LOADER = os.path.join(VARIABLES_FOLDER, 'train_loader.pkl')
    train_loader = joblib.load(TRAIN_LOADER)
    if RECOMPUTE is True:
        raise NameError('Forced recomputing values.')
except:
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, drop_last=True)
    joblib.dump(train_loader, TRAIN_LOADER)
try:
    TEST_LOADER = os.path.join(VARIABLES_FOLDER, 'test_loader.pkl')
    test_loader = joblib.load(TEST_LOADER)
    if RECOMPUTE is True:
        raise NameError('Forced recomputing values.')
except:
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=4, drop_last=True)
    joblib.dump(test_loader, TEST_LOADER)

try:
    VALID_LOADER = os.path.join(VARIABLES_FOLDER, 'valid_loader.pkl')
    valid_loader = joblib.load(VALID_LOADER)
    if RECOMPUTE is True:
        raise NameError('Forced recomputing values.')
except:
    valid_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, num_workers=4, drop_last=True)
    joblib.dump(valid_loader, VALID_LOADER)

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a model
model = LSTM(input_size=39, hidden_size=16, output_size=7, num_layers=3,
             bidirectional=False, dropout=0)

print(f'Model Parameters: {model.count_parameters(model)}')

# move model weights to device
model.to(DEVICE)
print(model)

criterion = torch.nn.CrossEntropyLoss()
parameters = model.parameters()
optimizer = optim.Adam(parameters)

#############################################################################
# Training Pipeline
#############################################################################
total_train_losses = []
total_test_losses = []
early_stopping = False

print(next(iter(train_loader)))

_model, t_loss, v_loss = fit(model, EPOCHS, lr=0.1,
                            loader=train_loader, v_loader=valid_loader,
                            earlyStopping=False)

modelname = os.path.join(VARIABLES_FOLDER, f'{_model.__class__.__name__}.pkl')
# Save model for later use
joblib.dump(_model, modelname)


#Final requested results
print_results(model, valid_loader, test_loader, t_loss, v_loss, EPOCHS)

