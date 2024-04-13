import os

import lightning as L
import torch
import torchvision
import wandb
from data_loader.CIFAR10 import CIFAR10
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import WandbLogger
from model.lightning_model import LightningModel
from model.resnet import resnet32


def create_model():
    model = torchvision.models.resnet18(weights=False, num_classes=10)
    model.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = torch.nn.Identity()
    return model


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(7)
    wandb.login()
    wandb_logger = WandbLogger(project="test_0314", name="resnet32,epoch=200")
    cifar10_dm = CIFAR10(num_workers=11, batch_size=256)
    cifar10_dm.prepare_data()
    cifar10_dm.setup()
    model = resnet32()
    lightning_model = LightningModel(model=model, batch_size=256, learning_rate=0.05)
    trainer = L.Trainer(
        max_epochs=200,
        accelerator="gpu",
        num_nodes=1,
        logger=wandb_logger,
    )
    trainer.fit(lightning_model, cifar10_dm)
    trainer.test(lightning_model, datamodule=cifar10_dm)
    wandb.finish()
