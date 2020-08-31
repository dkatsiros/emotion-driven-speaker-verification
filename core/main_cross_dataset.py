from utils.cross_dataset import CrossDataset as CrossDatasetClass
from utils import load_dataset
from dataloading.cross_dataset import CrossDataset
from torch.utils.data import DataLoader

# Constants
iemocap_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
emodb_mapping = {"N": 0, "W": 1, "F": 2, "T": 3}
ravdess_mapping = {0: 0, 5: 1, 3: 2, 4: 3}

# Create CrossDataset
cross_dataset = CrossDatasetClass()

# Init empty lists
X = []
y = []

# Add train dataset
X, y = load_dataset.load_IEMOCAP(train_only=True, n_classes=4)

assert cross_dataset.add_dataset_train(
    X=X, y=y, name="iemocap", mapping=iemocap_mapping)

X, y = load_dataset.load_Emodb(train_only=True)

assert cross_dataset.add_dataset_train(
    X=X, y=y, name="emodb", mapping=emodb_mapping)

# Get train data
X_train, y_train, X_val, y_val = cross_dataset.get_train_val_data(
    validation_ratio=.2)

# Add test dataset
X, y = load_dataset.load_RAVDESS(train_only=True)

assert cross_dataset.add_dataset_test(
    X=X, y=y, name="ravdess", mapping=ravdess_mapping)

# Get test data
X_test, y_test = cross_dataset.get_test_data(shuffle=False)


# Compute max sequence length -- bias here!!
# from scripts.find_max_seq_length import max_sequence_length
# max_seq_len = max_sequence_length(X=X_train+X_test+X_val)
max_seq_len = 320
print(f"Max sequence length: {max_seq_len}")


# Create datasets
FEATURE_EXTRACTION_METHOD = "MEL_SPECTROGRAM"

train_set = CrossDataset(X=X_train,
                         y=y_train,
                         feature_extraction_method=FEATURE_EXTRACTION_METHOD,
                         oversampling=True,
                         max_sequence_length=max_seq_len)

val_set = CrossDataset(X=X_val,
                       y=y_val,
                       feature_extraction_method=FEATURE_EXTRACTION_METHOD,
                       oversampling=True,
                       max_sequence_length=max_seq_len)

test_set = CrossDataset(X=X_test,
                        y=y_test,
                        feature_extraction_method=FEATURE_EXTRACTION_METHOD,
                        oversampling=True,
                        max_sequence_length=max_seq_len)


# Create dataloaders
BATCH_SIZE = 16
NUM_WORKERS = 4

train_loader = DataLoader(
    dataset=train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

val_loader = DataLoader(
    dataset=val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

test_loader = DataLoader(
    dataset=test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
