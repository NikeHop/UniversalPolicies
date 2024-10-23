import argparse
import os
import random

import gymnasium as gym
import torch 
import wandb
import yaml

from collections import defaultdict

from gymnasium.envs.registration import register
from minigrid.core.actions import ActionSpace
from minigrid.core.constants import COLOR_NAMES
from minigrid.wrappers import RGBImgObsWrapper, FullyObsWrapper
from transformers import AutoTokenizer, T5EncoderModel
from torchvision.io import write_video

from diffusion_nl.imitation_learning.model import ImitationPolicy
from diffusion_nl.diffusion_model.utils import state2img
from diffusion_nl.utils.utils import set_seed
from diffusion_nl.utils.environments import FIXINSTGOTO_ENVS

def eval(config,agent=None):

    # Load policy
    if agent is None:
        agent = ImitationPolicy.load_from_checkpoint(config["checkpoint"],map_location=config["device"])
        agent.eval()
    agent = agent.to(config["device"])
    
    # Load embedding model
    tokenizer = AutoTokenizer.from_pretrained(config["encoder_model"])
    encoder_model = T5EncoderModel.from_pretrained(config["encoder_model"])
    agent.load_embedding_model(encoder_model,tokenizer,config["device"])

    # Create environment
    action_space = ActionSpace(config["action_space"])
    if config["env"] == "goto":
        env_names = [random.choice(FIXINSTGOTO_ENVS) for _ in range(config["num_envs"])]
        envs = [
            FullyObsWrapper(
                gym.make(
                    f"{env_name}",
                    action_space=action_space,
                    num_dists=config["num_distractors"],
                )
            )
            for env_name in env_names
        ]

    elif config["env"] == "bosslevel":
        env_name = config["env_name"]
        envs = [
            FullyObsWrapper(
                gym.make(
                    env_name,
                    action_space=action_space,
                    num_dists=config["num_distractors"],
                )
            )
            for _ in range(config["num_envs"])
        ]

    total_rewards = defaultdict(list)
    completion_rate = defaultdict(list)
    trajectories = defaultdict(list)
    agent_ids = torch.tensor([config["action_space"] for _ in range(config["num_envs"])],dtype=torch.long)
    
    for _ in range(config["evaluation_episodes"]):
        # Reset the environments
        all_done = False
        obss = []

        for env in envs:
            obs, _ = env.reset()
            obss.append(obs)

        rewards = []

        for i in range(config["num_envs"]):
            trajectories[i].append(obss[i])
        n_timesteps = 0

        # Environment
        while not all_done:

            # Agent plans the next goals
            actions = agent.act(obss,agent_ids,config["device"])  # BxTxHxWxC

            new_obss = []  
            dones = []
            for i, (action, env) in enumerate(zip(actions, envs)):
                obs, reward, done, _, _ = env.step(action)
                dones.append(done)
                new_obss.append(obs)
                completion_rate[i].append(done)
                total_rewards[i].append(reward)
                trajectories[i].append(obs)

            obss = new_obss
            n_timesteps += 1

            # Check whether we are done
            all_done = all(dones)
            if n_timesteps > config["max_timesteps"]:
                all_done = True

    # Log statistics
    assert (config["num_envs"] * config["evaluation_episodes"]) > 0, "No data recorded"

    rewards = []
    completions = []
    for i in range(config["num_envs"]):
        rewards.append(max(total_rewards[i]))
        completions.append(any(completion_rate[i]))

    print(rewards, completions)

    wandb.log(
        {
            "average_reward": sum(rewards) / len(rewards),
            "completion_rate": sum(completions) / len(completions),
        }
    )

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


if __name__=="__main__":
    register_envs("diffusion_nl.environments.babyai.goto_specific:GoToSpecificObject")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,required=True)
    parser.add_argument("--device",type=int,default=None)
    parser.add_argument("--action_space",type=int,default=None)
    parser.add_argument("--checkpoint",type=str,default=None)
    parser.add_argument("--experiment_name",type=str,default=None)

    args = parser.parse_args()

    with open(args.config,"r") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    set_seed(config["seed"])

    if args.device is not None:
        config["device"] = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    if args.action_space is not None:
        config["action_space"] = args.action_space
    if args.checkpoint is not None:
        config["checkpoint"] = args.checkpoint
    if args.experiment_name is not None:
        config["logging"]["experiment_name"] = args.experiment_name

    wandb.init(
        project=config["logging"]["project"],
        name=config["logging"]["experiment_name"],
    )
    wandb.config.update(config)

    with torch.no_grad():
        eval(config)
