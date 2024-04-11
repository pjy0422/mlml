import lightning as L
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms


class CIFAR100(L.LightningDataModule):
    def __init__(
        self,
        data_path="./",
        batch_size: int = 64,
        num_workers: int = 0,
        height_width: tuple = (32, 32),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.num_workers = num_workers
        self.height_width = height_width

    def prepare_data(self):
        datasets.CIFAR100(root=self.data_path, download=True)
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(self.height_width),
                transforms.ToTensor(),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize(self.height_width),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage: str = None):
        train = datasets.CIFAR100(
            root=self.data_path,
            train=True,
            transform=self.train_transform,
            download=False,
        )
        self.test = datasets.CIFAR100(
            root=self.data_path,
            train=False,
            transform=self.test_transform,
            download=False,
        )
        self.train, self.val = random_split(train, [45000, 5000])

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
        return test_loader
