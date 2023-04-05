class Dataset:

    # default constructor
    def __init__(self, df, train_len, features):
        self.dataset_train = df.loc[: train_len - 1, features]
        self.dataset_test = df.loc[train_len:, features]
        self.mean = self.dataset_train.mean()
        self.std = self.dataset_train.std()

    def get_dataset(self):
        self.dataset_train = (self.dataset_train - self.mean) / self.std
        self.dataset_train = self.dataset_train.to_numpy()
        self.dataset_test = (self.dataset_test - self.mean) / self.std
        self.dataset_test = self.dataset_test.to_numpy()
        return self.dataset_train, self.dataset_test

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std
