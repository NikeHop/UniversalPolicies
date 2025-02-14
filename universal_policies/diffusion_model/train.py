"""
Training pipeline for standard diffusion model
"""

import argparse
import datetime
import os
import yaml

import gymnasium as gym
import pytorch_lightning as pl
import torch
import wandb

from gymnasium.envs.registration import register

from minigrid.wrappers import FullyObsWrapper
from minigrid.core.constants import COLOR_NAMES
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import isolate_rng

from universal_policies.diffusion_model.data import get_data
from universal_policies.diffusion_model.model import EDMModel
from universal_policies.diffusion_model.utils import get_eval_config
from universal_policies.diffuser_agent.eval import eval_ivd
from universal_policies.utils.utils import set_seed


def train(config):
    # Set seed
    with isolate_rng(include_cuda=True):
        # Create model directory
        model_directory = os.path.join(
            config["logging"]["model_directory"], config["logging"]["experiment_name"]
        )
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        # Set up image directory
        ct = datetime.datetime.now()
        image_directory = os.path.join(
            config["model"]["image_directory"],
            config["logging"]["experiment_name"],
            ct.strftime("%Y_%m_%d_%H_%M_%S"),
        )
        if not os.path.exists(image_directory):
            os.makedirs(image_directory)

        config["model"]["image_directory"] = image_directory

        # Set up Logger
        logger = WandbLogger(
            project=config["logging"]["project"], save_dir=model_directory
        )

        # Get Data
        (
            train_dataloader,
            test_dataloader,
            instruction2embed,
            example_contexts,
        ) = get_data(config)

        # Subsample example contexts with active action spaces
        filtered_example_contexts = {}
        for action_space in config["data"]["action_spaces"]:
            filtered_example_contexts[action_space] = example_contexts[action_space]

        # Create environment for evaluation of Model Performance
        env = get_env(config["env"])

        # Create Model
        model_class = get_model_class(config["model"]["model_type"])
        if config["model"]["load"]["load"]:
            model = model_class.load_from_checkpoint(
                config["model"]["load"]["checkpoint"]
            )
        else:
            model = model_class(env, config["model"])
        model.load_embeddings(instruction2embed)
        model.load_examples(filtered_example_contexts)

        # Create Trainer
        model_directory = os.path.join(
            config["logging"]["model_directory"],
            config["logging"]["experiment_name"],
            logger.name,
            wandb.run.id,
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
                val_check_interval=config["training"]["val_check_interval"],
                check_val_every_n_epoch=None,
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

        # Evaluate model
        with torch.no_grad():
            if config["evaluation"]["eval"]:
                for action_space in config["data"]["action_spaces"]:
                    with open(config["evaluation"]["ivd_config_file"], "r") as f:
                        ivd_config = yaml.safe_load(f)
                    eval_config = get_eval_config(
                        config, action_space, model, ivd_config
                    )

                    eval_ivd(eval_config)


def get_model_class(model_type):
    if model_type == "edm":
        return EDMModel
    else:
        raise NotImplementedError("Model type not implemented")


def get_env(config):
    if "GoToObjMaze" in config["env_name"]:
        env = gym.make(config["env_name"], num_dists=config["num_distractors"])
        env = FullyObsWrapper(env)

    elif "GoToObj" in config["env_name"]:
        if config["num_distractors"] > 0:
            env = gym.make(
                f"BabyAI-FixInstGoToPurpleBox-v0",
                num_dists=config["num_distractors"],
            )
            env = FullyObsWrapper(env)
        else:
            env = gym.make(config["env_name"])
            env = FullyObsWrapper(env)

    else:
        raise NotImplementedError("Environment not implemented")

    return env


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
    register_envs(
        "diffusion_nl.environments.babyai.goto_specific:GoToSpecificObject"
    )  # registers fixed instruction envs with gym

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
    parser.add_argument("--datapath", type=str, default=None, help="Path to dataset")

    parser.add_argument(
        "--w", type=float, default=None, help="Guiding strength for the diffusion model"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Number of updates for the model"
    )

    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate for the model"
    )
    parser.add_argument(
        "--action_space", type=int, default=None, help="Action spaces for the model"
    )
    parser.add_argument(
        "--conditional_prob",
        type=float,
        default=None,
        help="Probability of conditioning",
    )
    parser.add_argument(
        "--example_path", type=str, default=None, help="Path to example contexts"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint to continue from"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["seed"] = args.seed
    set_seed(args.seed)

    if args.experiment_name != None:
        config["logging"]["experiment_name"] = args.experiment_name

    if args.datapath != None:
        config["data"]["datapath"] = args.datapath

    if config["training"]["distributed"] and args.device != None:
        config["training"]["gpus"] = [args.device]
        config["evaluation"]["device"] = f"cuda:{args.device}"

    if args.max_steps != None:
        config["training"]["max_steps"] = args.max_steps

    if args.w != None:
        config["model"]["cond_w"] = args.w

    if args.lr != None:
        config["model"]["lr"] = args.lr

    if args.action_space != None:
        config["model"]["action_spaces"] = [args.action_space]
        config["data"]["action_spaces"] = [args.action_space]

    if args.conditional_prob != None:
        config["model"]["conditional_prob"] = args.conditional_prob

    if args.example_path != None:
        config["data"]["example_path"] = args.example_path

    if args.checkpoint != None:
        config["model"]["load"]["load"] = True
        config["model"]["load"]["checkpoint"] = args.checkpoint

    wandb.init(
        project=config["logging"]["project"], name=config["logging"]["experiment_name"]
    )
    wandb.config.update(config)

    train(config)
