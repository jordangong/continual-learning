from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


def get_default_parser(*args, **kwargs):
    parser = ArgumentParser(*args, **kwargs)
    parser.add_argument("--version", default=None, type=str)
    parser.add_argument("--log_path", default="lightning_logs", type=str)
    parser.add_argument("--wandb", default=False, action="store_true")
    parser.add_argument("--resume_ckpt_path", default=None, type=str)
    parser.add_argument("--track_grad", default=False, action="store_true")

    return parser


def get_logger(args, name=None):
    logger = [TensorBoardLogger(save_dir="lightning_logs",
                                name=name,
                                version=args.version)]
    if args.wandb and not args.fast_dev_run:
        wandb_logger = WandbLogger(save_dir="lightning_logs",
                                   name=name,
                                   version=args.version,
                                   project="continual-learning")
        logger.append(wandb_logger)

    return logger


def get_trainer(args, name=None, monitor="loss/val"):
    logger = get_logger(args, name=name)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(monitor=monitor, save_last=True, save_top_k=5)
    callbacks = [lr_monitor, model_checkpoint]
    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        limit_val_batches=1. if args.val_split > 0. else 0.,
        devices=args.devices if args.devices > 0 else None,
        num_nodes=args.num_nodes,
        accelerator="auto" if not args.fast_dev_run else "cpu",
        strategy="ddp_find_unused_parameters_false",
        sync_batchnorm=True if args.devices > 1 else False,
        precision=16 if args.fp16 else 32,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
    )

    return trainer
