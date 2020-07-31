from utils.cross_dataset import CrossDataset
from utils import load_dataset

# Constants
iemocap_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
emodb_mapping = {"N": 0, "W": 1, "F": 2, "T": 3}

# Create CrossDataset
cross_dataset = CrossDataset()

# Init empty lists
X = []
y = []

# Add train dataset
X, y = load_dataset.load_IEMOCAP(train_only=True, n_classes=4)

assert cross_dataset.add_dataset_train(
    X=X, y=y, name="iemocap", mapping=iemocap_mapping)

# Add test dataset
X, y = load_dataset.load_Emodb(train_only=True)

assert cross_dataset.add_dataset_test(
    X=X, y=y, name="emodb", mapping=emodb_mapping)

# print(cross_dataset.y_train, cross_dataset.y_test)

# Get train data
X_train, y_train, X_val, y_val = cross_dataset.get_train_val_data(
    validation_ratio=.2)
