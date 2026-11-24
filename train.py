import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_config
from dataset import SatMapDataset, graph_collate_fn
from model import SAMRoadplus
# import wandb
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor

parser = ArgumentParser()
parser.add_argument(
    "--config",
    default='',
    help="config file (.yml) containing the hyper-parameters for training. "
    "If None, use the nnU-Net config. See /config for examples.",
)
parser.add_argument(
    "--resume", default='', help="checkpoint of the last epoch of the model "
)
parser.add_argument(
    "--precision", default="16-mixed", help="32 or 16-mixed"
    )
parser.add_argument(
    "--fast_dev_run", default=False, action='store_true'
)
parser.add_argument(
    "--dev_run", default=False, action='store_true'
)


if __name__ == "__main__":
    print(f"[PID {os.getpid()}] Starting train.py", flush=True)
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"[PID {os.getpid()}] Config loaded", flush=True)
    dev_run = args.dev_run or args.fast_dev_run
    
    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="",
    #     # track hyperparameters and run metadata
    #     config=config,
    #     # disable wandb if debugging
    #     mode='disabled' # if dev_run else None
    # )
    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    net = SAMRoadplus(config)

    print(f"[PID {os.getpid()}] Loading training dataset", flush=True)
    train_ds = SatMapDataset(config, split='train', dev_run=dev_run)
    print(f"[PID {os.getpid()}] Training dataset loaded: {len(train_ds)} samples", flush=True)

    print(f"[PID {os.getpid()}] Loading validation dataset", flush=True)
    val_ds = SatMapDataset(config, split='val', dev_run=dev_run)
    print(f"[PID {os.getpid()}] Validation dataset loaded: {len(val_ds)} samples", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False, 
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )
    checkpoint_callback = ModelCheckpoint(dirpath="/local/s3425606/samroad_experiment/checkpoints", every_n_epochs=1, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # wandb_logger = WandbLogger()   
    logger = CSVLogger(save_dir="/data/samroad_experiment/logs", name="samroadplus")
    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        accelerator="gpu",
        devices=1, # was 2, but I changed it after the error
        # strategy="ddp", # For more than one device,
        strategy="auto", # For one device,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor],
        # logger=wandb_logger,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
        gradient_clip_val=1.0,
    )
    print(f"[PID {os.getpid()}] Starting trainer.fit()", flush=True)
    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume if args.resume else None)
    # trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)