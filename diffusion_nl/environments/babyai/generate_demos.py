import argparse
import logging
import multiprocessing as mp
import os
import pickle
import random
import time

from collections import defaultdict

import blosc
import gymnasium as gym
import matplotlib.pyplot as plt
from minigrid.core.actions import ActionSpace, Actions, ActionSpace
from minigrid.utils.baby_ai_bot_bfs import BabyAIBot
from minigrid.wrappers import RGBImgObsWrapper, FullyObsWrapper
import numpy as np
import torch
import wandb
import yaml

from tqdm import tqdm

from diffusion_nl.diffusion_model.utils import (
    transform_sample,
    extract_agent_pos_and_direction,
    state2img,
)
from diffusion_nl.utils.environments import FIXINSTGOTO_ENVS
from diffusion_nl.utils.utils import set_seed

from goto_specific import register_envs


def generate_episode(
    env_name: str,
    action_space: ActionSpace,
    minimum_length: int,
    max_steps: int,
    seed: int,
    use_img_obs: bool,
    num_distractors: int,
    debug: bool,
    use_agent_type: bool,
    img_size: int,
    resample_on_fail: bool = True,
):
    """
    Generate a single episode of demonstrations from the BabyAIBot

    Args:
        env_name (str): the name of the environment (without prefixes and suffixes)
        action_space (ActionSpace): action space
        minimum_length (int): minimum length of the episode (otherwise it is discarded)
        seed (int): seed
        use_img_obs (bool): if true uses image observation otherwise uses fully obs state space
        debug (bool): debug flag
        img_size (int): size of the image, only relevant if use_img_obs is true
        resample_on_fail (bool): if true resamples new seed to create new traj otherwise raises error
    """

    set_seed(seed)

    if "FixInstGoTo" in env_name:
        env = gym.make(
            env_name,
            highlight=False,
            action_space=action_space,
            num_dists=num_distractors,
            action_space_agent_color=use_agent_type,
        )
    elif "GoToObjMaze" in env_name:
        env = gym.make(
            env_name,
            highlight=False,
            action_space=action_space,
            num_dists=num_distractors,
            action_space_agent_color=use_agent_type,
        )
    elif "BossLevel" in env_name:
        env = gym.make(
            env_name,
            highlight=False,
            action_space=action_space,
            action_space_agent_color=use_agent_type,
        )
    else:
        raise NotImplementedError(
            f"This environment has not been implemented {env_name}"
        )

    # Apply Wrapper (image or state as obs)
    if use_img_obs:
        env = RGBImgObsWrapper(env, tile_size=img_size // env.unwrapped.width)
    else:
        env = FullyObsWrapper(env)

    mission_success = False
    curr_seed = seed

    # keep trying until we get a successful episode
    while not mission_success:
        try:
            done = False

            obs = env.reset(seed=curr_seed)[0]
            agent = BabyAIBot(env, action_space_type=action_space)

            actions = []
            mission = obs["mission"]

            images = []
            directions = []
            rewards = []
            n_steps = 0

            while not done:
                action = agent.replan()
                if isinstance(action, torch.Tensor):
                    action = action.item()

                new_obs, reward, done, _, _ = env.step(action)

                if done and reward > 0:
                    mission_success = True

                actions.append(action)
                obs_array = obs["image"]

                images.append(obs_array)
                directions.append(obs["direction"])
                rewards.append(reward)

                obs = new_obs
                n_steps += 1

                if n_steps > max_steps:
                    break

            # If our demos was successful, save it
            if mission_success:
                if len(images) >= minimum_length:
                    images.append(new_obs["image"])
                    return_mission = mission
                    return_tuple = (
                        return_mission,
                        env_name,
                        blosc.pack_array(np.array(images)),
                        directions,
                        actions,
                        rewards,
                        action_space,
                    )

                    return return_tuple
                else:
                    mission_success = False
                    curr_seed += random.randint(10, 100)
                    print("Mission not long enough. Resampling seed")
                    continue

            # Handle unsuccessful demos
            else:
                print(f"Mission unsuccessful. Seed {seed}")
                raise ValueError("Mission unsuccessful")

        except ValueError as e:
            if resample_on_fail:
                curr_seed += random.randint(10, 100)
                print(f"New_seed {curr_seed}")
                continue
            else:
                raise ValueError("Mission unsuccessful")


def generate_demos(
    n_episodes: int,
    num_workers: int,
    data_dir: str,
    env: str,
    env_name: str,
    action_space: ActionSpace,
    minimum_length: int,
    max_steps: int,
    seed: int,
    use_img_obs: bool,
    num_distractors: int,
    debug: bool,
    use_agent_type: bool = False,
    img_size: int = None,
    resample_on_fail: bool = True,
    save_data: bool = True,
    part: int = -1,
):
    
    all_demos = []
    for e in envs:
        demos = generate_demos_env(
            n_episodes,
            num_workers,
            data_dir,
            env,
            e,
            action_space,
            minimum_length=config["minimum_length"],
            max_steps=config["max_steps"],
            seed=config["seed"],
            img_size=config["img_size"],
            use_img_obs=config["use_img_obs"],
            num_distractors=config["num_distractors"],
            use_agent_type=config["use_agent_type"],
            part=config["part"],
        )

        all_demos += demos

    if config["part"] == -1:
        dataset_name = f"{config['action_space'].name}_{config['n_episodes']}_{config['minimum_length']}_{config['num_distractors']}_{config['use_agent_type']}_demos"
    else:
        dataset_name = f"{config['action_space'].name}_{config['n_episodes']}_{config['minimum_length']}_{config['num_distractors']}_{config['use_agent_type']}_demos_{config['part']}"

    filepath = os.path.join(
        config["data_directory"],
        config["env"],
        dataset_name,
        f"dataset_{config['n_episodes']}.pkl",
    )

    with open(filepath, "wb") as file:
        pickle.dump(all_demos, file)


def generate_demos_env(
    n_episodes: int,
    num_workers: int,
    data_dir: str,
    env: str,
    env_name: str,
    action_space: ActionSpace,
    minimum_length: int,
    max_steps: int,
    seed: int,
    use_img_obs: bool,
    num_distractors: int,
    debug: bool,
    use_agent_type: bool = False,
    img_size: int = None,
    resample_on_fail: bool = True,
    save_data: bool = True,
    part: int = -1,
):
    """
    Generate a set of agent demonstrations from the BabyAIBot

    Args:
        n_episodes (int): number of episodes to generate
        num_workers: number of workers to use for multiprocessing
        data_dir (str): path where generated data should be stored in baby_ai subdir
        env_name (str): the name of the environment (without prefixes and suffixes)
        action_space (ActionSpace): action space
        minimum_length (int): minimum length of the episode (otherwise it is discarded)
        seed (int): starting seed
        use_img_obs (bool): if true uses image observation otherwise uses fully obs state space
        debug (bool): debug flag
        img_size (int): size of the image, only relevant if use_img_obs is true
        resample_on_fail (bool): if true resamples new seed to create new traj otherwise raises error
    """

    checkpoint_time = time.time()

    demos = []

    if part == -1:
        constant = 0
    else:
        constant = (2 * n_episodes) * part

    seeds = range(seed + constant, seed + constant + 2 * n_episodes)
    print("Generated seeds")
    print(mp.cpu_count())
    pool = mp.Pool(processes=mp.cpu_count())

    n_generated_episodes = 0
    pbar = tqdm(total=n_episodes)
    batch_size = 10000
    while n_generated_episodes < n_episodes:
        pool = mp.Pool(processes=mp.cpu_count())
        results = [
            pool.apply_async(
                generate_episode,
                args=(
                    env_name,
                    action_space,
                    minimum_length,
                    max_steps,
                    seed,
                    use_img_obs,
                    num_distractors,
                    debug,
                    use_agent_type,
                    img_size,
                    resample_on_fail,
                ),
            )
            for seed in seeds[n_generated_episodes : n_generated_episodes + batch_size]
        ]

        for p in results:
            try:
                demo = p.get(timeout=60)
                demos.append(demo)
                n_generated_episodes += 1
                pbar.update()
                wandb.log({"n_episodes": n_generated_episodes})
            except Exception as e:
                print("Timeout")
                continue

        pool.close()

    demos = demos[:n_episodes]

    # log how long it took
    now = time.time()
    total_time = now - checkpoint_time
    print(f"total_time: {total_time}")

    # Save last batch of demos
    print("Saving demos...")
    if part == -1:
        demos_path = f"{data_dir}/{env}/{action_space.name}_{n_episodes}_{minimum_length}_{num_distractors}_{use_agent_type}_demos"
    else:
        demos_path = f"{data_dir}/{env}/{action_space.name}_{n_episodes}_{minimum_length}_{num_distractors}_{use_agent_type}_demos_{part}"

    if not os.path.exists(demos_path):
        os.makedirs(demos_path)

    if debug:
        demos_path = demos_path + f"_debug.pkl"
    elif env_name:
        demos_path = demos_path + f"/{env_name}.pkl"

    with open(demos_path, "wb+") as file:
        pickle.dump(demos, file)

    return demos


def check_subsequence(subsequence, action_space):
    if len(subsequence) < 3:
        return False
    print([int(elem) for elem in subsequence])
    if action_space == 0:
        # It should contain either a turn left or turn right
        subsequence = [int(elem) for elem in subsequence]
        action_types = set(subsequence)
        if 1 in action_types or 0 in action_types:
            return True

    elif action_space == 1:
        # It should contain three right turns
        subsequence = [int(elem) for elem in subsequence]
        action_types = set(subsequence)
        if 1 in action_types and len(action_types) == 1:
            return True

    elif action_space == 2:
        # It should contain three left turns
        subsequence = [int(elem) for elem in subsequence]
        action_types = set(subsequence)
        if 0 in action_types and len(action_types) == 1:
            return True

    elif action_space == 3:
        # It should contain a diagonal
        subsequence = [int(elem) for elem in subsequence]
        action_types = set(subsequence)
        print(action_types)
        if 7 in action_types or 8 in action_types:
            return True

    return False


def check_condition(examples, n_examples):
    print("Check conditions")
    if len(examples) < 4:
        return False

    for i, value in examples.items():
        print(i, len(value))
        if len(value) < n_examples:
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, help="Task to perform", required=True)
    parser.add_argument(
        "--config", type=str, help="Path to the config file", required=True
    )

    parser.add_argument(
        "--visualize", action="store_true", help="Visualize each action of the agent"
    )

    parser.add_argument(
        "--action_space",
        type=int,
        default=None,
        help="Type of action space the agent can choose",
    )

    parser.add_argument("--img_obs", action="store_true", help="Use image observations")
    parser.add_argument(
        "--n_episodes", type=int, help="How many episodes per environment to create"
    )
    parser.add_argument(
        "--n_dists",
        type=int,
        default=None,
        help="Number of distractors in the environment",
    )
    parser.add_argument(
        "--part",
        type=int,
        default=None,
        help="Which part of the dataset to generate",
    )
    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(logging.ERROR)

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    wandb.init(
        project=config["logging"]["project"], name=config["logging"]["experiment_name"]
    )
    wandb.config.update(config)

    if args.n_episodes != None:
        config["n_episodes"] = args.n_episodes

    if args.n_dists != None:
        config["num_distractors"] = args.n_dists

    if args.part != None:
        config["part"] = args.part

    # Pick an environment
    if config["env"] == "GOTO":
        envs = FIXINSTGOTO_ENVS
    elif config["env"] == "GOTO_LARGE":
        envs = ["BabyAI-GoToObjMazeS7-v0"]
    elif config["env"] == "BOSSLEVEL":
        envs = ["BabyAI-BossLevel-v0"]
    else:
        raise NotImplementedError(
            f"This environment has not been implemented {config['env']}"
        )

    # Register necessary envs with gym, not yet part of the register
    register_envs()
    if args.action_space != None:
        config["action_space"] = ActionSpace(args.action_space)
    else:
        config["action_space"] = ActionSpace(config["action_space"])

    generate_demos(
        n_episodes=config["n_episodes"],
        num_workers=config["num_workers"],
        data_dir=config["data_directory"],
        env=config["env"],
        env_names=envs,
        action_space=config["action_space"],
        minimum_length=config["minimum_length"],
        max_steps=config["max_steps"],
        seed=config["seed"],
        img_size=config["img_size"],
        use_img_obs=config["use_img_obs"],
        num_distractors=config["num_distractors"],
        use_agent_type=config["use_agent_type"],
        part=config["part"],
    )
