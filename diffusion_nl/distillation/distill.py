import argparse
import datetime
import os

import pytorch_lightning as pl
import wandb
import yaml

from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import isolate_rng

from diffusion_nl.utils.utils import set_seed
from diffusion_nl.distillation.model import DistillationModel
from diffusion_nl.diffusion_model.model import StateSpaceDiffusionModel
from diffusion_nl.diffusion_model.data import get_data


def distill(config):
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
        train_dataloader, test_dataloader, isntruction2label, example_contexts = (
            get_data(config)
        )

        # Load the teacher model to distill
        teacher = get_teacher(config["model_store"], config["teacher"], config["device"])

        # Create Model
        if config["model"]["load"]["load"]:
            model = DistillationModel.load_from_checkpoint(
                config["model"]["load"]["checkpoint"]
            )
        else:
            model = DistillationModel(teacher, config["model"])


        # Create Trainer
        model_directory = os.path.join(
            config["logging"]["model_directory"], config["logging"]["experiment_name"]
        )

        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        callbacks = []
        if config["training"]["early_stopping"]:
            callbacks.append(
                EarlyStopping(
                    monitor="validation/loss",
                    min_delta=0.00,
                    patience=3,
                    verbose=False,
                    mode="min",
                )
            )

        if config["training"]["save_all"]:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=model_directory,
                    save_top_k=-1,
                )
            )

        if config["training"]["distributed"]:
            trainer = pl.Trainer(
                default_root_dir=model_directory,
                max_steps=config["training"]["max_steps"],
                logger=logger,
                accelerator=config["training"]["accelerator"],
                devices=config["training"]["gpus"],
                strategy=config["training"]["strategy"],
                check_val_every_n_epoch=config["training"]["check_val_every_n_epoch"],
                callbacks=callbacks,
            )

        else:
            trainer = pl.Trainer(
                default_root_dir=model_directory,
                max_steps=config["training"]["max_steps"],
                logger=logger,
                accelerator=config["training"]["accelerator"],
                callbacks=callbacks,
            )

        # Training
        trainer.fit(model, train_dataloader, test_dataloader)



def get_teacher(model_store_path,config,device):
    model_path = os.path.join(model_store_path,config["path"], config["name"], config["checkpoint"])
    model = StateSpaceDiffusionModel.load_from_checkpoint(model_path,map_location=device)
    return model


if __name__ == "__main__":
    """
    register_envs(
        "diffusion_nl.environments.goto_specific:GoToSpecificObject"
    )  # registers fixed instruction envs with gym
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42, help="Seed for reproducibility"
    )
    parser.add_argument(
        "-d", "--device", type=int, default=None, help="Device to train on"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment"
    )
    parser.add_argument("--data_path", type=str, default=None, help="Path to dataset")

    parser.add_argument(
        "--w", type=float, default=None, help="Guiding strength for the diffusion model"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Number of updates for the model"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["seed"] = args.seed
    set_seed(args.seed)

    if args.experiment_name != None:
        config["logging"]["experiment_name"] = args.experiment_name

    if args.data_path != None:
        config["data"]["data_path"] = args.data_path

    if config["training"]["distributed"] and args.device != None:
        config["training"]["gpus"] = [args.device]

    if args.max_steps != None:
        config["training"]["max_steps"] = args.max_steps

    if args.w != None:
        config["model"]["cond_w"] = args.w

    wandb.init(
        mode=config["logging"]["mode"],
        project=config["logging"]["project"],
        name=config["logging"]["experiment_name"],
    )
    wandb.config.update(config)

    distill(config)
