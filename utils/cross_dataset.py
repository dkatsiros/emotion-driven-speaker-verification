from sklearn.model_selection import StratifiedShuffleSplit


class CrossDataset():
    def __init__(self):
        super().__init__()

        # Store train data
        self.X_train = []  # a list of files
        self.y_train = []  # a list of integers/classes
        self.names_of_datasets_train = []
        self.X_idx_train = []
        # Store test data
        self.X_test = []
        self.y_test = []
        self.names_of_datasets_test = []
        self.X_idx_test = []

    def add_dataset_train(self, X, y, name=None, mapping=None):
        """Add a dataset to the CrossDataset training.
### Arguments:

        X {list} : A list of sample paths.

        y {list} : A list of labels.

        name {str} : Datasets name.

        mapping {dict} : A dictionary mappin each label to
                         `Neutral:0`, `Anger:1`,
                         `Happiness:2` and
                         `Sadness:3` respectively.
### Returns:

        True if the dataset was added else False.

"""
        # Check everything
        if X == [] or y == [] or name is None or mapping is None:
            print(
                'Dataset insertion error. Please check that all arguments were correct.')
            return False

        # Add dataset name
        if name in self.names_of_datasets_train:
            print(f'Dataset is already imported.')
            return False

        X, y = self.remove_extra_labels(X=X, y=y, mapping=mapping)

        # Get dataset index. Starting from 0.
        dataset_idx = len(self.names_of_datasets_train)
        self.names_of_datasets_train.append(name)

        # Add samples
        self.X_train += X

        # Add idx: [dtst_idx,dtst_idx, ...., dtst_idx]
        self.X_idx_train += [dataset_idx]*len(X)

        # Add y
        self.y_train += [int(mapping[l]) for l in y]

        return True

    def add_dataset_test(self, X, y, name=None, mapping=None):
        """Add a dataset to the CrossDataset testing.
### Arguments:

        X {list} : A list of sample paths.

        y {list} : A list of labels.

        name {str} : Datasets name.

        mapping {dict} : A dictionary mappin each label to
                         `Neutral:0`, `Anger:1`,
                         `Happiness:2` and
                         `Sadness:3` respectively.
### Returns:

        True if the dataset was added else False.

"""
        # Check everything
        if X == [] or y == [] or name is None or mapping is None:
            print(
                'Dataset insertion error. Please check that all arguments were correct.')
            return False

        # Add dataset name
        if name in self.names_of_datasets_test:
            print(f'Dataset is already imported.')
            return False

        X, y = self.remove_extra_labels(X=X, y=y, mapping=mapping)

        # Get dataset index. Starting from 0.
        dataset_idx = len(self.names_of_datasets_test)
        self.names_of_datasets_test.append(name)

        # Add samples
        self.X_test += X

        # Add idx: [dtst_idx,dtst_idx, ...., dtst_idx]
        self.X_idx_test += [dataset_idx]*len(X)

        # Add y
        self.y_test += [int(mapping[l]) for l in y]

        return True

    def get_train_val_data(self, validation_ratio=.2):
        """Return a split in X_train with ratio `validation_ratio`.

### Arguments:

        validation_ratio {float} : The ration of train/validation.

### Returns:

        X_train {list} : Training data.

        y_train {list} : Training labels.

        X_val {list} : Validation data.

        y_val {list} : Validation labels.
        """

        # Initialize
        X_train, y_train = [], []
        X_val, y_val = [], []

        # Split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio)
        train_idx, val_idx = next(
            sss.split(self.X_train, self.y_train))

        # Train
        for idx in train_idx:
            X_train.append(self.X_train[idx])
            y_train.append(self.y_train[idx])

        # Validation
        for idx in val_idx:
            X_val.append(self.X_train[idx])
            y_val.append(self.y_train[idx])

        return X_train, y_train, X_val, y_val

    @staticmethod
    def remove_extra_labels(X, y, mapping):
        """For a given mapping, remove each y that doesn't exist in that.
### Arguments:
        X {list}: A list of samples.

        y {list}: A list of labels.

        mapping {dict}: A mapping from labels to integers
                         that correspond to cross-dataset's
                         labels.

### Returns:
        X_new {list}: A list with all samples that are
                        left after filtering.

        y_new {list}: A list with all labels that are
                        left after filtering.
        """

        X_new = []
        y_new = []
        keys = list(mapping.keys())

        for x_, y_ in zip(X, y):
            if y_ not in keys:
                continue
            X_new.append(x_)
            y_new.append(y_)
        return X_new, y_new
