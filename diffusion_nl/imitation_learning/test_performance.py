import argparse
import gymnasium as gym
import yaml
import os
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities.seed import isolate_rng
import wandb
from minigrid.wrappers import RGBImgObsWrapper, FullyObsWrapper
from diffusion_nl.imitation_learning.data import get_data
from diffusion_nl.imitation_learning.model import Imit
from diffusion_nl.environments.goto_specific import register_envs

import argparse

from collections import defaultdict

import gymnasium as gym
import yaml
import wandb

from minigrid.core.actions import NamedActionSpace
from minigrid.wrappers import FullyObsWrapper
from torchvision.io import write_video
from diffusion_nl.utils.utils import set_seed
from diffusion_nl.diffusion_model.utils import state2img


def eval(config):
    # Load evaluation environments
    ids = register_envs(num_dists=3)
    action_space = NamedActionSpace(config["action_space"])
    envs = [
        FullyObsWrapper(gym.make(ids[i % len(ids)], action_space=action_space)) for i in range(config["num_envs"])
    ]

    # Load the agent
    agent = get_agent(config)
    total_rewards = []
    completion_rate = []


    for _ in range(config["evaluation_episodes"]):
        for i, env in enumerate(envs):
            rewards = []
            dones = []


            n_timesteps = 0
            done = False
            obs, _ = env.reset()
            while not done:
                action = agent.act([obs])
                obs, reward, done, _, _ = env.step(action)
                rewards.append(reward)
                dones.append(done)
                n_timesteps += 1

                if n_timesteps > config["max_timesteps"]:
                    done = True
            total_rewards.append(max(rewards))
            completion_rate.append(max(rewards) > 0)




    assert (config["num_envs"] * config["evaluation_episodes"]) > 0, "No data recorded"
    print(sum(total_rewards)/len(total_rewards), sum(completion_rate)/len(completion_rate))
    wandb.log(
        {
            "average_reward": sum(total_rewards)
                              / len(total_rewards),
            "completion_rate": sum(completion_rate)
                               / len(completion_rate),
        }
    )

def get_agent(config):
    if config["agent_type"] == "imitation":
        train_dataloader, test_dataloader, instruction2label = get_data(config)
        print(len(instruction2label))
        model = Imit.load_from_checkpoint(os.path.join(config["model_store"], config["model_path"], config["model_name"], config["model_checkpoint"]))
        model.inst2label = instruction2label
        return model

    else:
        raise NotImplementedError(f"Agent type {config['agent_type']} not implemented")






if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluation script for the diffuser agent"
    )
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--dm_model_checkpoint", type=str, help="Checkpoint to load")
    parser.add_argument("--dm_model_name", type=str, help="Model name")
    parser.add_argument("--device", type=str, help="Which device to use")

    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)

    if args.dm_model_checkpoint!=None:
        config["dm_model_checkpoint"] = args.dm_model_checkpoint
    if args.dm_model_name!=None:
        config["dm_model_name"] = args.dm_model_name
    if args.device!=None:
        config["device"] = args.device

    wandb.init(
        project=config["logging"]["project"],
        name=config["logging"]["experiment_name"],
    )
    wandb.config.update(config)

    eval(config)

