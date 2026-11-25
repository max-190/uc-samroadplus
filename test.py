import os
from argparse import ArgumentParser

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from dataset import SatMapDataset, graph_collate_fn
from model import SAMRoadplus
from utils import load_config

os.environ['CUDA_VISIBLE_DEVICES'] = ''

parser = ArgumentParser()
parser.add_argument(
    "--config",
    default='',
    help="config file (.yml) containing the hyper-parameters for training. "
    "If None, use the nnU-Net config. See /config for examples.",
)
parser.add_argument(
    "--checkpoint", default='', help="checkpoint of the model to test."
)
parser.add_argument(
    "--precision", default="16-mixed", help="32 or 16"
)


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    net = SAMRoadplus(config)

    val_ds = SatMapDataset(config, is_train=False, dev_run=False)

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )

    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor],
        # strategy='ddp_find_unused_parameters_true',
        precision=args.precision,
        # profiler=profiler
        )
    trainer.test(net, dataloaders=val_loader, ckpt_path=args.checkpoint)