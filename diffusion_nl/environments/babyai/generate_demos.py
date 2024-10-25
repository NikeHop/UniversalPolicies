import argparse
import logging
import multiprocessing as mp
import os
import pickle
import random

import blosc
import gymnasium as gym
import numpy as np
import torch
import wandb
import yaml

from minigrid.core.actions import ActionSpace, ActionSpace
from minigrid.utils.baby_ai_bot_bfs import BabyAIBot
from minigrid.wrappers import RGBImgObsWrapper, FullyObsWrapper
from tqdm import tqdm

from diffusion_nl.environments.babyai.goto_specific import register_envs, FIXINSTGOTO_ENVS
from diffusion_nl.utils.utils import set_seed



def generate_episode(
    env_name: str,
    action_space: ActionSpace,
    minimum_length: int,
    max_steps: int,
    seed: int,
    use_img_obs: bool,
    num_distractors: int,
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
        max_steps (int): maximum number of steps possible in the episode
        seed (int): seed
        use_img_obs (bool): if true uses image observation otherwise uses fully obs state space
        num_distractors (int): number of distractors in the environment
        use_agent_type (bool): if true give the agent a colour depending on its type
        img_size (int): size of the image, only relevant if use_img_obs is true
        resample_on_fail (bool): if true resamples new seed to create new traj otherwise raises error
    """

    set_seed(seed)

    env = gym.make(
        env_name,
        highlight=False,
        action_space=action_space,
        num_dists=num_distractors,
        action_space_agent_color=use_agent_type,
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
    action_space: ActionSpace,
    minimum_length: int,
    max_steps: int,
    seed: int,
    img_size: int,
    use_img_obs: bool,
    num_distractors: int,
    use_agent_type: bool = False,
    resample_on_fail: bool = True,
    part: int = -1,
):
    # Pick an environment
    if env == "GOTO" or env == "GOTO_DISTRACTORS":
        envs = FIXINSTGOTO_ENVS
    elif env == "GOTO_DISTRACTORS_LARGE":
        envs = ["BabyAI-GoToObjMazeS7-v0"]
    else:
        raise NotImplementedError(
            f"This environment has not been implemented {env}"
        )

    all_demos = []
    for e in envs:
        demos = generate_demos_env(
            n_episodes,
            num_workers,
            e,
            action_space,
            minimum_length,
            max_steps,
            seed,
            use_img_obs,
            num_distractors,
            use_agent_type,
            img_size,
            resample_on_fail,
            part
        )

        all_demos += demos

    if part == -1:
        dataset_name = f"{action_space.name}_{n_episodes}_{minimum_length}_{num_distractors}_{use_agent_type}_demos"
    else:
        dataset_name = f"{action_space.name}_{n_episodes}_{minimum_length}_{num_distractors}_{use_agent_type}_demos_{part}"

    directory_path = os.path.join(data_dir, env, dataset_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    filepath = os.path.join(
        directory_path,
        f"dataset_{n_episodes}.pkl",
    )

    with open(filepath, "wb") as file:
        pickle.dump(all_demos, file)


def generate_demos_env(
    n_episodes: int,
    num_workers: int,
    env_name: str,
    action_space: ActionSpace,
    minimum_length: int,
    max_steps: int,
    seed: int,
    use_img_obs: bool,
    num_distractors: int,
    use_agent_type: bool = False,
    img_size: int = None,
    resample_on_fail: bool = True,
    part: int = -1,
):
    """
    Generate a set of agent demonstrations from the BabyAIBot

    Args:
        n_episodes (int): number of episodes to generate
        num_workers: number of workers to use for multiprocessing
        env_name (str): the name of the environment (without prefixes and suffixes)
        action_space (ActionSpace): action space
        minimum_length (int): minimum length of the episode (otherwise it is discarded)
        max_steps (int): maximum number of steps possible in the episode
        seed (int): starting seed
        use_img_obs (bool): if true uses image observation otherwise uses fully obs state space
        num_distractors (int): number of distractors in the environment
        use_agent_type (bool): if true give the agent a colour depending on its type
        img_size (int): size of the image, only relevant if use_img_obs is true
        resample_on_fail (bool): if true resamples new seed to create new traj otherwise raises error
        part (int): which part of the dataset to generate
    """

    # Make sure that different parts have different seeds
    if part == -1:
        constant = 0
    else:
        constant = (2 * n_episodes) * part

    seeds = range(seed + constant, seed + constant + 2 * n_episodes)
    
    if num_workers == -1:
        num_workers = mp.cpu_count()
    
    demos = []
    n_generated_episodes = 0
    pbar = tqdm(total=n_episodes)
    batch_size = 10000
    while n_generated_episodes < n_episodes:
        pool = mp.Pool(processes=num_workers)
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
                logging.error("Timeout")
                continue

        pool.close()

    demos = demos[:n_episodes]

    return demos



if __name__ == "__main__":
    parser = argparse.ArgumentParser()


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

    # Load config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Integrate CLI arguments
    if args.n_episodes != None:
        config["n_episodes"] = args.n_episodes

    if args.n_dists != None:
        config["num_distractors"] = args.n_dists

    if args.action_space != None:
        config["action_space"] = ActionSpace(args.action_space)
    else:
        config["action_space"] = ActionSpace(config["action_space"])
    
    # Set up logging
    logging.basicConfig()
    logging.getLogger().setLevel(logging.ERROR)
    wandb.init(
        mode=config["logging"]["mode"],
        project=config["logging"]["project"],
        name=config["logging"]["experiment_name"]
    )
    wandb.config.update(config)

    # Register necessary envs with gym, not yet part of the register
    register_envs()
    
    # Generate demos
    generate_demos(
        n_episodes=config["n_episodes"],
        num_workers=config["num_workers"],
        data_dir=config["data_directory"],
        env=config["env"],
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
