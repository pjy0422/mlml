import os

import hydra
import lightning as L
import torch
import torchvision
import wandb
from data_loader.CIFAR10 import CIFAR10
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import WandbLogger
from model.lightning_model import LightningModel
from model.resnet import resnet32
from omegaconf import DictConfig, OmegaConf


def create_model():
    model = torchvision.models.resnet18(weights=False, num_classes=10)
    model.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = torch.nn.Identity()
    return model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.to_yaml(cfg)
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(cfg.params.seed)
    wandb.login()
    wandb_logger = WandbLogger(
        project=cfg.params.wandb_name, name=cfg.params.wandb_name
    )
    cifar10_dm = CIFAR10(
        num_workers=cfg.params.num_workers, batch_size=cfg.params.batch_size
    )
    cifar10_dm.prepare_data()
    cifar10_dm.setup()

    lightning_model = LightningModel(cfg=cfg)
    lightning_model.load_pretrained_model()
    trainer = L.Trainer(
        max_epochs=cfg.params.max_epochs,
        accelerator=cfg.params.accelerator,
        num_nodes=cfg.params.num_nodes,
        logger=wandb_logger,
    )
    trainer.fit(lightning_model, cifar10_dm)
    trainer.test(lightning_model, datamodule=cifar10_dm)
    wandb.finish()


if __name__ == "__main__":
    main()
