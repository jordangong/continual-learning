from argparse import ArgumentParser, BooleanOptionalAction

import pytorch_lightning as pl
import torch.nn
from lightning_fabric import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import nn, optim
from torchmetrics import Accuracy
from torchvision.models import resnet18
from torchvision.transforms import transforms

from datamodule import CIFAR10DataModule
from scheduler import linear_warmup_decay


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

        model = resnet18(num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        model.maxpool = nn.Identity()
        self.model = model

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
        logits = self.model(img)
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
        parser.add_argument("--batch_size", default=512, type=int)
        parser.add_argument("--num_workers", default=2, type=int)
        parser.add_argument("--max_epochs", default=100, type=int)
        parser.add_argument("--max_steps", default=-1, type=int)
        parser.add_argument("--learning_rate", default=1e-3, type=float)
        parser.add_argument("--weight_decay", default=1e-2, type=float)
        parser.add_argument("--warmup_epochs", default=10, type=int)

        return parser


if __name__ == '__main__':
    seed_everything(0)
    parser = ArgumentParser()
    parser.add_argument("--version", default=None, type=str)
    parser.add_argument("--log_path", default="lightning_logs", type=str)
    parser.add_argument("--resume_ckpt_path", default=None, type=str)
    parser.add_argument("--track_grad", default=False, action=BooleanOptionalAction)
    parser = CIFARResNet.add_model_specific_args(parser)
    args = parser.parse_args()

    dm = CIFAR10DataModule(data_dir=args.data_dir, batch_size=args.batch_size,
                           num_workers=args.num_workers)
    args.num_samples = dm.num_samples
    args.num_classes = dm.num_classes

    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    dm.train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.ToTensor(),
        torch.jit.script(nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.2,
            ),
            transforms.RandomGrayscale(p=0.2),
            normalize,
        )),
    ])
    dm.val_transforms = dm.test_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        torch.jit.script(normalize),
    ])

    model = CIFARResNet(**args.__dict__)
    tensorboard_logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="pretrain",
        version=args.version
    )
    wandb_logger = WandbLogger(
        save_dir="lightning_logs",
        name="pretrain",
        version=args.version,
        project="continual-learning",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(monitor="loss/val",
                                       save_last=True, save_top_k=5)

    callbacks = [lr_monitor, model_checkpoint]
    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        devices=args.devices if args.devices > 0 else None,
        num_nodes=args.num_nodes,
        accelerator="gpu" if args.devices > 0 else None,
        strategy="ddp_find_unused_parameters_false",
        sync_batchnorm=True if args.devices > 1 else False,
        precision=16 if args.fp16 else 32,
        callbacks=callbacks,
        logger=[tensorboard_logger, wandb_logger],
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_ckpt_path)
