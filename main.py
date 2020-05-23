from utils.load_dataset import load_Emodb
from torch.utils.data import DataLoader


# PyTorch
BATCH_SIZE = 128
EPOCHS = 50


X_train, y_train, X_test, y_test = load_Emodb(train_split_percentage=0.3)

train_loader = DataLoader(X_train, batch_sampler=BATCH_SIZE, num_workers=4)
