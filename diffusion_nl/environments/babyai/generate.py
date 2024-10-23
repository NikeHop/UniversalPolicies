import argparse
import logging
import multiprocessing as mp
import os
import pickle
import random
import time
import sys

from collections import defaultdict
from copy import deepcopy
print(sys.path)

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

from torchvision.io import write_video
from tqdm import tqdm

from diffusion_nl.diffusion_model.utils import transform_sample, extract_agent_pos_and_direction, state2img
from diffusion_nl.utils.environments import (
    seeding,
    ENVS,
    TEST_ENV,
    CLASS2ENV,
    FIXINSTGOTO_ENVS,
)
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
    resample_on_fail: bool = True
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

    seeding(seed)
    # sample a random environment
    if debug:
        env_name = "BabyAI-GoToObj-v0"

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
        raise NotImplementedError(f"This environment has not been implemented {env_name}")

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
    part:int = -1
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

    if part==-1:
        constant=0
    else:
        constant = (2*n_episodes) * part

    seeds = range(seed+constant, seed+constant + 2*n_episodes)
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
                    resample_on_fail
                ),
            )
            for seed in seeds[n_generated_episodes:n_generated_episodes + batch_size]
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
    if part==-1:
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


GO_TO_IRRELEVANT_ACTIONS = [Actions(3), Actions(4), Actions(5), Actions(6)]
BOSSLEVEL_IRRELEVANT_ACTIONS = [Actions(6)]

def generate_example_trajectories(config):
    if config["trajectory_type"]=="full_action_space_single_random":
        generate_full_action_space_random(
            config["env"],
            config["seed"],
            config["use_img_obs"],
            config["num_distractors"],
            config["img_size"],
            config["n_action_spaces"],
            config["save_directory"],
            config["n_examples"]
        )
    elif config["trajectory_type"] == "every_action":
        generate_examples(config)
    else:
        raise NotImplementedError(f"This trajectory type has not been implemented {config['trajectory_type']}")

def generate_full_action_space_random(
    env_name: str,
    seed: int,
    use_img_obs: bool,
    num_distractors: int,
    img_size: int,
    n_action_spaces: int,
    save_directory: str,
    n_examples: int
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
    """
    # Seed everything
    seeding(seed)

    # Set up datastructure 
    example_contexts = defaultdict(list)

    # Iterate over the action spaces
    for _ in range(n_examples):
        for action_space in range(n_action_spaces):
            action_space = ActionSpace(action_space)
            legal_actions = action_space.get_legal_actions()

            if "GOTO"==env_name:
                # Sample an environment 
                env_id = random.choice(FIXINSTGOTO_ENVS)
                env = gym.make(
                    env_id,
                    highlight=False,
                    action_space=action_space,
                    num_dists=num_distractors,
                )
                irrelevant_actions = GO_TO_IRRELEVANT_ACTIONS
            elif "BossLevel" in env_name:
                env = gym.make(
                    env_name,
                    highlight=False,
                    action_space=action_space,
                )
                irrelevant_actions = BOSSLEVEL_IRRELEVANT_ACTIONS
            else:
                raise NotImplementedError(
                    f"This environment has not been implemented {env_name}"
                )

            # Apply Wrapper (image or state as obs)
            if use_img_obs:
                env = RGBImgObsWrapper(env, tile_size=img_size // env.unwrapped.width)
            else:
                env = FullyObsWrapper(env)

            success = False 
            while not success:
                obs = env.reset()[0]
                images = [obs["image"]]
                executed_actions = []
                while len(executed_actions)<len(legal_actions):
                    # Get possible actions at current state

                    possible_actions = get_possible_actions(obs["image"])
                    legal_possible_actions = [action for action in legal_actions if action in possible_actions and action not in executed_actions]

                    if len(legal_possible_actions)==0:
                        break

                    for action in legal_possible_actions:
                        if action in irrelevant_actions:
                            executed_actions.append(action)
                            continue

                        new_obs, reward, done, _, _ = env.step(action)
                        obs = new_obs
                        images.append(obs["image"])
                        executed_actions.append(action)

                
                if len(executed_actions)==len(legal_actions):
                    success = True
                    compressed_video = blosc.pack_array(np.stack(images,axis=0))
                    example_contexts[action_space].append(compressed_video)
                else:
                    seed += 1

    # Save example contexts
    filename = f"{env_name}_{n_examples}_{num_distractors}_full_action_space_random.pkl"
    with open(os.path.join(save_directory,filename),"wb") as file:
        pickle.dump(example_contexts,file)
    
    # Get longest episode
    for action_space, examples in example_contexts.items():
        longest_episode = max([blosc.unpack_array(example).shape[0] for example in examples])
        print(f"Action space {action_space} longest episode {longest_episode}")

def generate_examples(config):
    # Prepare data structure
    example_contexts = defaultdict(list)
    save_directory = config["save_directory"]

    # Generate a start state
    env_id = FIXINSTGOTO_ENVS[0]
    env = gym.make(
        env_id,
        highlight=False,
        action_space=ActionSpace.all,
        num_dists=0,
        action_space_agent_color=True,
    )

    env = FullyObsWrapper(env)
    _ = env.reset()[0]
    grid = env.grid
    cleaned_grid = []
    for elem in grid.grid:
        if isinstance(elem,minigrid.core.world_object.Wall):
            cleaned_grid.append(elem)
        elif elem is None:
            cleaned_grid.append(elem)
        else:
            cleaned_grid.append(None)

    grid.grid = cleaned_grid
    env.set_state(grid,(3,3),0,0)
    
    # All actions to test
    actions = list(range(4)) + list(range(7,19))
    for action_space in ActionSpace:
        legal_actions = action_space.get_legal_actions()
        grid.grid = cleaned_grid
        env.set_state(grid,(3,3),3,0)
        obs = env.step(6)
        state = obs[0]["image"]
        example_contexts[action_space].append(state)

        if action_space == ActionSpace.all:
            continue
        
        for action in actions:
            grid.grid = cleaned_grid
            env.set_state(grid,(3,3),3,0)
            if action not in legal_actions:
                action = 6
            obs = env.step(action)
            state = obs[0]["image"]
            example_contexts[action_space].append(state)
    
    # Format the contexts
    for action_space, examples in example_contexts.items():
        example_contexts[action_space] = [blosc.pack_array(np.stack(examples,axis=0))]

    # Save examples
    filename = f"no_padding_each_action_0.pkl"
    with open(os.path.join(save_directory, filename), "wb") as file:
        pickle.dump(example_contexts, file)


def get_possible_actions(obs):
    """
    Get the possible actions at the current state
    """
    possible_actions = []
    # extract agent position and direction
    (x, y), direction = extract_agent_pos_and_direction(obs)
    for action in Actions:
        if action == 0:
            # Left, always possible
            possible_actions.append(action)
        elif action == 1:
            # Right, always possible
            possible_actions.append(action)
        elif action == 2:
            # Forward, check if there is a wall in front
            if direction == 0:
                # Facing west
                if x <= 5:
                    possible_actions.append(action)
            elif direction == 1:
                # Facing south
                if y <= 5:
                    possible_actions.append(action)
            elif direction == 2:
                # Facing east
                if x >= 2:
                    possible_actions.append(action)
            elif direction == 3:
                # Facing north
                if y >= 2:
                    possible_actions.append(action)
        elif action == 3:
            # Pickup
            possible_actions.append(action)
        elif action == 4:
            # Drop
            possible_actions.append(action)
        elif action == 5:
            # Toggle
            possible_actions.append(action)
        elif action == 6:
            # Done
            possible_actions.append(action)
        elif action == 7:
            # Diagonal left
            if direction == 0:
                # Facing west
                if x <= 5 and y >= 2:
                    possible_actions.append(action)
            elif direction == 1:
                # Facing south
                if y <= 5 and x <= 5:
                    possible_actions.append(action)
            elif direction == 2:
                # Facing east
                if x >= 2 and y <= 5:
                    possible_actions.append(action)
            elif direction == 3:
                # Facing north
                if y >= 2 and x >= 2:
                    possible_actions.append(action)
        elif action == 8:
            # Diagonal right
            if direction == 0:
                # Facing west
                if x <= 5 and y <= 5:
                    possible_actions.append(action)
            elif direction == 1:
                # Facing south
                if y <= 5 and x >= 2:
                    possible_actions.append(action)
            elif direction == 2:
                # Facing east
                if x >= 2 and y >= 2:
                    possible_actions.append(action)
            elif direction == 3:
                # Facing north
                if y >= 2 and x <= 5:
                    possible_actions.append(action)
        elif action == 9 or action == 17:
            # Right move
            if x <= 5:
                possible_actions.append(action)
        elif action == 10:
            # Down move
            if y <= 5:
                possible_actions.append(action)
        elif action == 11 or action == 16:
            # Left move
            if x >= 2:
                possible_actions.append(action)
        elif action == 12:
            # Up move
            if y >= 2:
                possible_actions.append(action)
        elif action == 13:
            # Turn around, always possible
            possible_actions.append(action)
        elif action == 14:
            # Diagonal backwards left, same as diagonal left if we would turn 180 degrees
            if direction == 0:
                # Facing west
                if x >= 2 and y >= 2:
                    possible_actions.append(action)
            elif direction == 1:
                # Facing south
                if y >= 2 and x <= 5:
                    possible_actions.append(action)
            elif direction == 2:
                # Facing east
                if x <= 5 and y <= 5:
                    possible_actions.append(action)
            elif direction == 3:
                # Facing north
                if y <= 5 and x >= 2:
                    possible_actions.append(action)
        elif action == 15:
            # Diagonal backwards right, same as diagonal right if we would turn 180 degrees
            if direction == 0:
                # Facing west
                if x >= 2 and y <= 5:
                    possible_actions.append(action)
            elif direction == 1:
                # Facing south
                if y >= 2 and x >= 2:
                    possible_actions.append(action)
            elif direction == 2:
                # Facing east
                if x <= 5 and y >= 2:
                    possible_actions.append(action)
            elif direction == 3:
                # Facing north
                if y <= 5 and x <= 5:
                    possible_actions.append(action)
        elif action == 18:
            # Backward, same as forward if we would turn 180 degrees
            if direction == 2:
                # Facing east
                if x <= 5:
                    possible_actions.append(action)
            elif direction == 3:
                # Facing north
                if y <= 5:
                    possible_actions.append(action)
            elif direction == 0:
                # Facing west
                if x >= 2:
                    possible_actions.append(action)
            elif direction == 1:
                # Facing south
                if y >= 2:
                    possible_actions.append(action)

    return possible_actions


if __name__ == "__main__":
    register_envs()  # registers fixed instruction envs with gym

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, help="Task to perform", required=True)
    parser.add_argument(
        "--config", type=str, help="Path to the config file", required=True
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to generate the debug dataset or not",
    )
    parser.add_argument(
        "--test", action="store_true", help="Generates one episode for each environment"
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
    if config["env"]=="GOTO":
        envs = FIXINSTGOTO_ENVS
    
    elif config["env"]=="GOTO_LARGE":
        envs = ["BabyAI-GoToObjMazeS7-v0"]
        
    elif config["env"]=="BOSSLEVEL":
        envs = ["BabyAI-BossLevel-v0"]
    else:
        raise NotImplementedError(f"This environment has not been implemented {config['env']}")

    if args.task == "generate_demos":

        if args.action_space != None:
            config["action_space"] = ActionSpace(args.action_space)
        else:
            config["action_space"] = ActionSpace(config["action_space"])

        if args.debug:
            generate_demos(
                n_episodes=config["n_episodes"],
                num_workers=config["num_workers"],
                data_dir=config["data_directory"],
                env_name="GoToLocal",
                action_space=config["action_space"],
                minimum_length=config["minimum_length"],
                seed=config["seed"],
                img_size=config["img_size"],
                debug=args.debug,
                use_img_obs=config["use_img_obs"],
            )
        else:
            all_demos = []
            for e in envs:
                demos = generate_demos(
                    n_episodes=config["n_episodes"],
                    num_workers=config["num_workers"],
                    data_dir=config["data_directory"],
                    env=config["env"],
                    env_name=e,
                    action_space=config["action_space"],
                    minimum_length=config["minimum_length"],
                    max_steps=config["max_steps"],
                    seed=config["seed"],
                    img_size=config["img_size"],
                    use_img_obs=config["use_img_obs"],
                    num_distractors=config["num_distractors"],
                    debug=args.debug,
                    use_agent_type=config["use_agent_type"],
                    part=config["part"]
                )

                all_demos += demos

            if config["part"]==-1:
                dataset_name = f"{config['action_space'].name}_{config['n_episodes']}_{config['minimum_length']}_{config['num_distractors']}_{config['use_agent_type']}_demos"
            else:
                dataset_name = f"{config['action_space'].name}_{config['n_episodes']}_{config['minimum_length']}_{config['num_distractors']}_{config['use_agent_type']}_demos_{config['part']}"
            
            print(dataset_name)
            print(len(all_demos))
            filepath = os.path.join(
                config["data_directory"],
                config["env"],
                dataset_name,
                f"dataset_{config['n_episodes']}.pkl",
            )
            with open(filepath, "wb") as file:
                pickle.dump(all_demos, file)

    elif args.task == "generate_example_trajectories":
        generate_example_trajectories(config)
    else:
        raise NotImplementedError(f"This task has not been implemented {args.task}")
