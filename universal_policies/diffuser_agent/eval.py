"""
Evaluation script for the diffuser agent 
"""

import argparse
import os
import random

from collections import defaultdict

import gymnasium as gym
import torch
import tqdm
import wandb
import yaml

from gymnasium.envs.registration import register

from minigrid.core.actions import ActionSpace
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.constants import COLOR_NAMES
from torchvision.io import write_video

from universal_policies.diffuser_agent.agent import (
    DeterministicAgent,
    RandomAgent,
    DiffusionAgent,
    LocalAgent,
)
from universal_policies.utils.utils import set_seed
from universal_policies.diffusion_model.utils import state2img
from universal_policies.environments.babyai.goto_specific import FIXINSTGOTO_ENVS


def eval(config):
    if config["action_model"] == "ivd":
        eval_ivd(config)
    elif config["action_model"] == "local_policy":
        eval_local_policy(config)
    else:
        raise NotImplementedError(
            f"Action model {config['action_model']} not implemented"
        )


def eval_ivd(config, model=None):
    # Create evaluation environments
    action_space = ActionSpace(config["action_space"])

    if config["env"] == "goto":
        env_names = [random.choice(FIXINSTGOTO_ENVS) for _ in range(config["num_envs"])]
        envs = [
            FullyObsWrapper(
                gym.make(
                    f"{env_name}",
                    action_space=action_space,
                    num_dists=config["num_distractors"],
                    action_space_agent_color=config["use_agent_type"],
                )
            )
            for env_name in env_names
        ]

    elif config["env"] == "goto_large":
        envs = [
            FullyObsWrapper(
                gym.make(
                    "BabyAI-GoToObjMazeS7-v0",
                    action_space=action_space,
                    num_dists=config["num_distractors"],
                )
            )
            for _ in range(config["num_envs"])
        ]

    else:
        raise NotImplementedError(f"Environment {config['env']} not implemented")

    # Load the agent
    agent = get_agent(config, model)
    total_rewards = defaultdict(list)
    completion_rate = defaultdict(list)
    trajectories = defaultdict(list)

    for _ in range(config["evaluation_episodes"]):
        # Reset the environments
        all_done = False
        obss = []
        agent.reset()

        for env in envs:
            obs, _ = env.reset()
            obss.append(obs)

        rewards = []
        dones = []
        for i in range(config["num_envs"]):
            trajectories[i].append(obss[i])

        pbar = tqdm.tqdm(total=config["max_timesteps"])
        n_timesteps = 0
        n_timesteps_prev = 0
        while not all_done:
            # Agent chooses actions
            batch_actions = agent.act(obss)

            # Perform actions
            new_obss = []
            for actions in batch_actions:
                for i, (action, env) in enumerate(zip(actions, envs)):
                    obs, reward, done, _, _ = env.step(action)
                    rewards.append(reward)
                    dones.append(done)
                    completion_rate[i].append(done)
                    total_rewards[i].append(reward)
                    new_obss.append(obs)
                    trajectories[i].append(obs)

                n_timesteps += 1

            # Check whether we are done
            all_done = all(dones)
            obss = new_obss[-config["num_envs"] :]

            if n_timesteps > config["max_timesteps"]:
                all_done = True

            pbar.update(n_timesteps - n_timesteps_prev)
            n_timesteps_prev = n_timesteps

        # Visualize the trajectories
        if config["visualize_traj"]:
            for i in range(min(10, config["num_envs"])):
                video = []

                for obs, done in zip(trajectories[i], [False] + completion_rate[i]):
                    img = state2img(obs["image"])
                    video.append(img)
                    instruction = obs["mission"]
                    if done:
                        break

                filename = f"{config['action_space']}_{instruction}_{i}.mp4"
                filedir = os.path.join(
                    config["model_store"],
                    config["dm_model_path"],
                    config["dm_model_name"],
                    "evaluation_videos",
                )

                if not os.path.exists(filedir):
                    os.makedirs(filedir)
                filepath = os.path.join(filedir, filename)

                write_video(filepath, video, fps=10)

    # Log statistics
    assert (config["num_envs"] * config["evaluation_episodes"]) > 0, "No data recorded"

    rewards = []
    completions = []
    for i in range(config["num_envs"]):
        rewards.append(max(total_rewards[i]))
        completions.append(any(completion_rate[i]))

    wandb.log(
        {
            "average_reward": sum(rewards) / len(rewards),
            "completion_rate": sum(completions) / len(completions),
        }
    )


def eval_local_policy(config):
    # Create evaluation environments
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

    # Load the agent
    agent = get_agent(config)
    total_rewards = defaultdict(list)
    completion_rate = defaultdict(list)
    trajectories = defaultdict(list)

    for _ in range(config["evaluation_episodes"]):
        # Reset the environments
        agent.reset()
        all_done = False
        obss = []

        for env in envs:
            obs, _ = env.reset()
            obss.append(obs)

        rewards = []
        dones = []
        for i in range(config["num_envs"]):
            trajectories[i].append(obss[i])

        n_timesteps = 0

        # Environment
        while not all_done:
            # Agent plans the next goals
            plans = agent.plan(obss)  # BxTxHxWxC

            # Perform actions
            new_obss = []

            # For every goal, execute k timesteps
            for i in range(config["n_frames"]):
                for k in range(config["n_timesteps"]):
                    actions = agent.act(obss, plans[:, i])
                    for n, (action, env) in enumerate(zip(actions, envs)):
                        obs, reward, done, _, _ = env.step(action)
                        rewards.append(reward)
                        dones.append(done)
                        completion_rate[n].append(done)
                        total_rewards[n].append(reward)
                        new_obss.append(obs)
                        trajectories[n].append(obs)

                    obss = new_obss[-config["num_envs"] :]
                    n_timesteps += 1

            # Check whether we are done
            all_done = all(dones)
            obss = new_obss[-config["num_envs"] :]

            if n_timesteps > config["max_timesteps"]:
                all_done = True

        # Visualize the trajectories
        if config["visualize"]:
            for i in range(min(10, config["num_envs"])):
                video = []
                for obs in trajectories[i]:
                    img = state2img(obs["image"])
                    video.append(img)
                    instruction = obs["mission"]

                filename = f"{instruction}_{i}.mp4"
                filedir = os.path.join(
                    config["model_store"],
                    config["dm_model_path"],
                    config["dm_model_name"],
                    "evaluation_videos",
                )

                if not os.path.exists(filedir):
                    os.makedirs(filedir)
                filepath = os.path.join(filedir, filename)

                write_video(filepath, video, fps=2)

    # Log statistics
    assert (config["num_envs"] * config["evaluation_episodes"]) > 0, "No data recorded"

    rewards = []
    completions = []
    for i in range(config["num_envs"]):
        rewards.append(max(total_rewards[i]))
        completions.append(any(completion_rate[i]))

    wandb.log(
        {
            "average_reward": sum(rewards) / len(rewards),
            "completion_rate": sum(completions) / len(completions),
        }
    )


def get_agent(config, model):
    if config["agent_type"] == "deterministic":
        return DeterministicAgent(config["action"])
    elif config["agent_type"] == "random":
        return RandomAgent(config["n_actions"])
    elif config["agent_type"] == "diffusion":
        if config["planning_type"] == "ivd":
            agent = DiffusionAgent(config)
            agent.load_examples(config["example_path"])
        elif config["planning_type"] == "local_policy":
            agent = LocalAgent(config)
            agent.load_examples(config["example_path"])
        return agent
    else:
        raise NotImplementedError(f"Agent type {config['agent_type']} not implemented")


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

    parser = argparse.ArgumentParser(
        description="Evaluation script for the diffuser agent"
    )

    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument(
        "--checkpoint", type=str, help="Checkpoint of the diffusion planner to load"
    )
    parser.add_argument("--device", type=str, help="Which device to use")
    parser.add_argument(
        "--action_space", type=int, default=None, help="action space for the agent"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment added to the name in the config file",
    )
    parser.add_argument(
        "--example_path", type=str, default=None, help="Path to the dataset of examples"
    )
    parser.add_argument(
        "--n_example_frames",
        type=int,
        default=None,
        help="Number of frames to use for the examples",
    )
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)

    if args.checkpoint != None:
        config["checkpoint"] = args.checkpoint
    if args.device != None:
        config["device"] = args.device
        config["embeddings"]["device"] = args.device
    if args.action_space != None:
        config["action_space"] = args.action_space
    if args.experiment_name != None:
        config["logging"]["experiment_name"] += args.experiment_name
    if args.example_path != None:
        config["example_path"] = args.example_path
    if args.n_example_frames != None:
        config["n_example_frames"] = args.n_example_frames

    wandb.init(
        project=config["logging"]["project"],
        name=config["logging"]["experiment_name"],
        mode=config["logging"]["mode"],
    )
    wandb.config.update(config)

    set_seed(args.seed)
    with torch.no_grad():
        eval(config)
