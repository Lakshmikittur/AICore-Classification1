import numpy
import torch
import torchvision



class DatasetXy(torch.utils.data.Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y), "Data and labels have to be of equal length!"
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    # Not dependent on index
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)


class DatasetX(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X)
    # Not dependent on index
    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return len(self.X)


class Dataloader:
    def __init__(self, args, train_data, test_data):
        self.args = args
        self.train_data = train_data
        self.test_data =  test_data

    def test(self):
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.args.batch_size,
            pin_memory=self.args.device == "cuda",
        )

    def train(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=self.args.device == "cuda",
        )







