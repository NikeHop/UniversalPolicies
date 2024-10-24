import argparse
import os
import yaml

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import isolate_rng
import wandb

from diffusion_nl.ivd_model.data import get_data
from diffusion_nl.ivd_model.model import IVDBabyAI
from diffusion_nl.utils.utils import set_seed

def train_ivd(config):
    with isolate_rng(include_cuda=True):
        # Setup directory for saving models
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
        train_dataloader, test_dataloader = get_data(config["data"])

        # Choose model 
        if config["data"]["dataset"] == "babyai":
            IVD = IVDBabyAI
            action_space = config["model"]["action_space"]
        else:
            raise NotImplementedError(f"Unknown model {config['data']['dataset']}")
        
        # Create Model
        if config["model"]["load"]["load"]:
            model = IVD.load_from_checkpoint(
                os.path.join(model_directory, config["model"]["load"]["checkpoint"])
            )
        else:
            model = IVD(config["model"]["lr"], action_space)

        # Create Trainer
        model_directory = os.path.join(
            config["logging"]["model_directory"], config["logging"]["experiment_name"]
        )

        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        if config["training"]["early_stopping"]:
            callbacks = [
                EarlyStopping(
                    monitor="validation/acc",
                    min_delta=0.00,
                    patience=3,
                    verbose=False,
                    mode="max",
                )
            ]
        else:
            callbacks = []

        if config["training"]["distributed"]:
            trainer = pl.Trainer(
                default_root_dir=model_directory,
                max_epochs=config["training"]["max_epochs"],
                logger=logger,
                accelerator=config["training"]["accelerator"],
                devices=config["training"]["gpus"],
                strategy=config["training"]["strategy"],
                callbacks=callbacks,
            )

        else:
            trainer = pl.Trainer(
                default_root_dir=model_directory,
                max_epochs=config["training"]["max_epochs"],
                logger=logger,
                accelerator=config["training"]["accelerator"],
                callbacks=callbacks,
            )
            
        # Training
        trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="path to experiment config file",
    )
    parser.add_argument(
        "--datapath", type=str, default=None, help="path to data file"
    )
    parser.add_argument(
        "--action_space", type=int, default=None, help="Action space to use"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "rb") as file:
        config = yaml.safe_load(file)

    # Seed everything
    set_seed(config["seed"])

    # Update config with CLI arguments
    if args.datapath is not None:
        config["data"]["datapath"] = args.datapath

    if args.action_space is not None:
        config["model"]["action_space"] = args.action_space

    # Setup wandb project
    wandb.init(
        mode=config["logging"]["mode"],
        project=config["logging"]["project"],
        name=config["logging"]["experiment_name"]
    )
    wandb.config.update(config)

    train_ivd(config)
