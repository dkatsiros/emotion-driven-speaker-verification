import torch
from torch import optim
from utils.load_dataset import load_Emodb
from dataloading import EmodbDataset
from torch.utils.data import DataLoader

from models.lstm import LSTM


# PyTorch
BATCH_SIZE = 128
EPOCHS = 50

# Split dataset to arrays
X_train, y_train, X_test, y_test, X_eval, y_eval = load_Emodb()
# Load sets using dataset class
train_set = EmodbDataset(X_train, y_train)
test_set = EmodbDataset(X_test, y_test)
eval_set = EmodbDataset(X_eval, y_eval)
# PyTorch DataLoader
train_loader = DataLoader(train_set, batch_sampler=BATCH_SIZE, num_workers=4)
test_loader = DataLoader(test_set,batch_sampler=BATCH_SIZE, num_workers=4)
eval_loader = DataLoader(eval_set,batch_sampler=BATCH_SIZE, num_workers=4)

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a model
model = LSTM(input_size=39, hidden_size=16, output_size=7, num_layers=1,
             bidirectional=False, dropout=0)

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

for epoch in range(1, EPOCHS + 1):
    pass