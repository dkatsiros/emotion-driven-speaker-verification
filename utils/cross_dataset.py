class CrossDataset:
    def __init__(self):
        super().__init__()
        # Store data

        self.X_train = []  # a list of files
        self.y_train = []  # a list of integers/classes
        self.names_of_datasets_train = []
        self.X_idx_train = []

        self.X_test = []
        self.y_test = []
        self.names_of_datasets_test = []
        self.X_idx_test = []

    def add_dataset_train(self, X, y, name=None, mapping=None):
        """Add a dataset to the CrossDataset training.
Arguments:

        X {list} : A list of sample paths.

        y {list} : A list of labels.

        name {str} : Datasets name.

        mapping {dict} : A dictionary mappin each label to
                         `Neutral:0`, `Anger:1`, 
                         `Happiness:2` and 
                         `Sadness:3` respectively.
Returns:

        True if the dataset was added else False.

"""
        # Check everything
        if X == [] or y == [] or name is None or mapping is None:
            print(
                'Dataset insertion error. Please check that all arguments were correct.')
            return False
        # Add dataset name
        if name in self.name_of_datasets:
            print(f'Dataset is already imported.')
            return False
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
Arguments:

        X {list} : A list of sample paths.

        y {list} : A list of labels.

        name {str} : Datasets name.

        mapping {dict} : A dictionary mappin each label to
                         `Neutral:0`, `Anger:1`, 
                         `Happiness:2` and 
                         `Sadness:3` respectively.
Returns:

        True if the dataset was added else False.

"""
        # Check everything
        if X == [] or y == [] or name is None or mapping is None:
            print(
                'Dataset insertion error. Please check that all arguments were correct.')
            return False
        # Add dataset name
        if name in self.name_of_datasets_test:
            print(f'Dataset is already imported.')
            return False
        # Get dataset index. Starting from 0.
        dataset_idx = len(self.names_of_datasets_test)
        self.names_of_datasets_test.append(name)

        # Add samples
        self.X += X_test

        # Add idx: [dtst_idx,dtst_idx, ...., dtst_idx]
        self.X_idx_test += [dataset_idx]*len(X)

        # Add y
        self.y_test += [int(mapping[l]) for l in y]

        return True
