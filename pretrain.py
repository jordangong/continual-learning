import copy
from argparse import ArgumentParser

import pytorch_lightning as pl
from lightning_fabric import seed_everything
from torch import nn, optim
from torchmetrics import Accuracy
from torchvision.models import resnet18

from datamodule import cifar10aug
from scheduler import linear_warmup_decay
from utils import get_default_parser, get_trainer


class CIFARResNet(pl.LightningModule):
    def __init__(
            self,
            num_nodes,
            devices,
            num_samples,
            num_classes,
            batch_size,
            max_epochs,
            learning_rate,
            weight_decay,
            warmup_epochs,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        global_batch_size = num_nodes * devices * batch_size \
            if devices > 0 else batch_size
        self.train_iters_per_epoch = num_samples // global_batch_size

        encoder = resnet18(num_classes=num_classes)
        encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        encoder.maxpool = nn.Identity()
        self.encoder = encoder
        self.classifiers = nn.ModuleList([copy.deepcopy(encoder.fc)])
        self.encoder.fc = nn.Identity()

        self.loss = nn.CrossEntropyLoss()

        self.train_acc1 = Accuracy("multiclass", num_classes=num_classes, top_k=1)
        self.train_acc5 = Accuracy("multiclass", num_classes=num_classes, top_k=5)
        self.val_acc1 = Accuracy("multiclass", num_classes=num_classes, top_k=1,
                                 computer_on_step=False)
        self.val_acc5 = Accuracy("multiclass", num_classes=num_classes, top_k=5,
                                 computer_on_step=False)
        self.test_acc1 = Accuracy("multiclass", num_classes=num_classes, top_k=1,
                                  computer_on_step=False)
        self.test_acc5 = Accuracy("multiclass", num_classes=num_classes, top_k=5,
                                  computer_on_step=False)

    def shared_step(self, batch, acc1, acc5):
        img, target = batch
        logits = self.classifiers[0](self.encoder(img))
        loss = self.loss(logits, target)
        acc1 = acc1(logits.softmax(-1), target)
        acc5 = acc5(logits.softmax(-1), target)

        return loss, acc1, acc5

    def training_step(self, batch, batch_idx):
        loss, acc1, acc5 = self.shared_step(batch, self.train_acc1, self.train_acc5)

        self.log_dict({
            "loss/train": loss,
            "acc/train/top1": acc1,
            "acc/train/top5": acc5,
        }, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc1, acc5 = self.shared_step(batch, self.val_acc1, self.val_acc5)

        self.log_dict({
            "loss/val": loss,
            "hp_metric": acc1,
            "acc/val/top1": acc1,
            "acc/val/top5": acc5,
        }, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc1, acc5 = self.shared_step(batch, self.test_acc1, self.test_acc5)

        self.log_dict({
            "acc/test/top1": acc1,
            "acc/test/top5": acc5,
        }, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate,
                                weight_decay=self.weight_decay)

        warmup_iters = self.train_iters_per_epoch * self.warmup_epochs
        total_iters = self.train_iters_per_epoch * self.max_epochs
        scheduler = {
            "scheduler": optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_iters, total_iters)
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--num_nodes", default=1, type=int)
        parser.add_argument("--devices", default=1, type=int)
        parser.add_argument("--fast_dev_run", default=False, type=int)
        parser.add_argument("--fp16", default=False, action="store_true")
        parser.add_argument("--data_dir", default="data", type=str)
        parser.add_argument("--val_split", default=0.2, type=float)
        parser.add_argument("--batch_size", default=192, type=int)
        parser.add_argument("--num_workers", default=2, type=int)
        parser.add_argument("--max_epochs", default=200, type=int)
        parser.add_argument("--max_steps", default=-1, type=int)
        parser.add_argument("--learning_rate", default=5e-3, type=float)
        parser.add_argument("--weight_decay", default=3e-2, type=float)
        parser.add_argument("--warmup_epochs", default=20, type=int)

        return parser


if __name__ == '__main__':
    seed_everything(0)
    parser = CIFARResNet.add_model_specific_args(get_default_parser())
    args = parser.parse_args()

    dm = cifar10aug(args)
    model = CIFARResNet(**args.__dict__)
    trainer = get_trainer(args, name="pretrain")

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_ckpt_path)
    if args.val_split == 0:
        trainer.test(datamodule=dm, ckpt_path="best")
