import argparse
import os
import yaml

import lightning.pytorch as pl
import wandb

from gymnasium.envs.registration import register
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities.seed import isolate_rng
from minigrid.core.constants import COLOR_NAMES

from universal_policies.imitation_learning.data import get_data
from universal_policies.imitation_learning.model import ImitationPolicy
from universal_policies.imitation_learning.eval import eval
from universal_policies.utils.utils import set_seed


def train_imitation_learning(config):
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
        train_dataloader, test_dataloader = get_data(config)

        # Create Model
        if config["model"]["load"]["load"]:
            model = ImitationPolicy.load_from_checkpoint(
                config["model"]["load"]["checkpoint"],
                map_location=config["embeddings"]["device"],
            )
        else:
            model = ImitationPolicy(config)

        # Create Trainer
        if config["training"]["distributed"]:
            trainer = pl.Trainer(
                default_root_dir=model_directory,
                logger=logger,
                accelerator=config["training"]["accelerator"],
                devices=config["training"]["gpus"],
                strategy=config["training"]["strategy"],
                max_epochs=config["training"]["max_epochs"],
            )

        else:
            trainer = pl.Trainer(
                default_root_dir=model_directory,
                logger=logger,
                accelerator=config["training"]["accelerator"],
                max_epochs=config["training"]["max_epochs"],
            )

        # Training
        trainer.fit(model, train_dataloader, test_dataloader)

        # Evaluation
        if config["eval"]["eval"]:
            eval(config["eval"], model)


def register_envs(entrypoint):
    """
    Registers envs with all combinations of objects and colors to gym
    """
    for color in COLOR_NAMES:
        for obj in ["ball", "box", "key"]:
            print("Register environment")
            register(
                id=f"BabyAI-FixInstGoTo{color.capitalize()}{obj.capitalize()}-v0",
                entry_point=entrypoint,
                kwargs={"room_size": 8, "num_dists": 7, "color": color, "obj": obj},
            )


if __name__ == "__main__":
    register_envs("diffusion_nl.environments.babyai.goto_specific:GoToSpecificObject")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="path to experiment config file",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--max_gap",
        type=int,
        default=None,
        help="maximum timegap between observation and goal",
    )
    parser.add_argument("--action_space", type=int, default=None, help="Action space")
    parser.add_argument(
        "--datapath", type=str, default=None, help="Path to training dataset"
    )
    parser.add_argument(
        "--experiment_name", type=str, default="", help="Name of the experiment"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to run the model on"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint to load"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "rb") as file:
        config = yaml.safe_load(file)

    # Update config with CLI arguments
    if args.seed is not None:
        config["seed"] = args.seed

    if args.max_gap is not None:
        config["data"]["max_gap"] = args.max_gap

    if args.action_space is not None:
        config["model"]["action_space"] = args.action_space
        config["eval"]["action_space"] = args.action_space

    if args.datapath is not None:
        config["data"]["datapath"] = args.datapath

    if args.device is not None:
        config["training"]["gpus"] = [int(args.device)]
        config["eval"]["device"] = f"cuda:{args.device}"
        config["embeddings"]["device"] = f"cuda:{args.device}"

    if args.checkpoint is not None:
        config["model"]["load"]["load"] = True
        config["model"]["load"]["checkpoint"] = args.checkpoint

    # Setup wandb project
    wandb.init(
        project=config["logging"]["project"],
        mode=config["logging"]["mode"],
        tags=config["logging"]["tags"],
        name=config["logging"]["experiment_name"] + args.experiment_name,
    )

    wandb.config.update(config)

    set_seed(config["seed"])

    train_imitation_learning(config)
