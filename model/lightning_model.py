import os

import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import random_split
import torchmetrics
from model.resnet import resnet32
from torch.optim.lr_scheduler import OneCycleLR
from torchattacks import PGDL2


model_path = "./epoch=199-step=70200.ckpt"
class LightningModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.load_model()
        self.learning_rate = cfg.params.learning_rate
        self.num_classes = cfg.params.num_classes
        self.batch_size = cfg.params.batch_size
        self.save_hyperparameters()
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.weight_decay = cfg.params.weight_decay
        self.momentum = cfg.params.momentum
        self.optimizer = cfg.params.optimizer
        self.pretrained_model_path = model_path
    def load_model(self):
        Model = resnet32()
        self.model = Model
    def load_pretrained_model(self):
        self.pretrained_model = LightningModel.load_from_checkpoint("./epoch=199-step=70200.ckpt",map_location='cpu')
        # eval mode
        self.pretrained_model.eval()
    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        ##### training with adversarial examples #####
        features, true_labels = batch
        attack = PGDL2(model = self.pretrained_model, eps = 1.0, alpha = 0.2, steps = 10, random_start = True)
        adv_features = attack(features, true_labels)        
        logits = self(adv_features)
        loss = F.cross_entropy(logits, true_labels) + F.cross_entropy(self(features), true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        ##### original code #####
        #loss, true_labels, predicted_labels = self._shared_step((features, true_labels))
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
            momentum=self.momentum,
            weight_decay=self.weight_decay,
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
