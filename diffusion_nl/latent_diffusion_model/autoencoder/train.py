"""
Training pipeline for AutoEncoder
"""

import argparse
import datetime
import os
import yaml

import gymnasium as gym
import pytorch_lightning as pl
import wandb

from minigrid.wrappers import FullyObsWrapper
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities.seed import isolate_rng

from diffusion_nl.latent_diffusion_model.autoencoder.data import get_data
from diffusion_nl.latent_diffusion_model.autoencoder.model import Autoencoder


def train(config):
    # Set seed
    with isolate_rng(include_cuda=True):
        # Create model directory
        model_directory = os.path.join(
            config["logging"]["model_directory"], config["logging"]["experiment_name"]
        )
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        # Set up Logger
        logger = WandbLogger(
            project=config["logging"]["project"], save_dir=model_directory
        )

        # Get Data
        train_dataloader, test_dataloader = get_data(config)

        # Create Model
        if config["model"]["load"]["load"]:
            model = Autoencoder.load_from_checkpoint(
                os.path.join(model_directory, config["model"]["load"]["checkpoint"])
            )
        else:
            model = Autoencoder(config["model"])

        # Create Trainer
        model_directory = os.path.join(
            config["logging"]["model_directory"], config["logging"]["experiment_name"]
        )

        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        if config["training"]["distributed"]:
            trainer = pl.Trainer(
                default_root_dir=model_directory,
                max_steps=config["training"]["max_steps"],
                logger=logger,
                accelerator=config["training"]["accelerator"],
                devices=config["training"]["gpus"],
                strategy=config["training"]["strategy"]
            )

        else:
            trainer = pl.Trainer(
                default_root_dir=model_directory,
                max_steps=config["training"]["max_steps"],
                logger=logger,
                accelerator=config["training"]["accelerator"],
            )

        # Training
        trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    wandb.init(
        project=config["logging"]["project"], name=config["logging"]["experiment_name"]
    )

    train(config)
