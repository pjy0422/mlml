import os

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import OneCycleLR


class LightningModel(L.LightningModule):
    def __init__(
        self,
        model,
        learning_rate: float = 0.1,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        optimizer: str = "sgd",
        num_classes: int = 10,
        batch_size: int = 64,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.save_hyperparameters(ignore=["model"])
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // self.batch_size
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
