"""
Code to generate the example trajectories for conditioning the planner
"""

import argparse
import os
import pickle
import random

from collections import defaultdict

import blosc
import gymnasium as gym
import minigrid
import numpy as np
import yaml

from minigrid.core.actions import ActionSpace, ActionSpace, Actions
from minigrid.wrappers import FullyObsWrapper

from diffusion_nl.diffusion_model.utils import extract_agent_pos_and_direction
from diffusion_nl.environments.babyai.goto_specific import (
    register_envs,
    FIXINSTGOTO_ENVS,
)
from diffusion_nl.utils.utils import set_seed

GO_TO_IRRELEVANT_ACTIONS = [Actions(3), Actions(4), Actions(5), Actions(6)]


def generate_example_trajectories(config):
    if config["example_type"] == "action_space_random":
        generate_action_space_random(
            config["env"],
            config["seed"],
            config["num_distractors"],
            config["n_examples"],
            config["save_directory"],
        )
    elif config["example_type"] == "slot_per_action":
        generate_slot_per_action(config["env"], config["save_directory"])
    else:
        raise NotImplementedError(
            f"This trajectory type has not been implemented {config['trajectory_type']}"
        )


def generate_slot_per_action(env: str, save_directory: str):
    """
    Generate example trajectories by executing each action in the action space

    Args:
        save_directory: Directory to save the examples to.
    """

    # Prepare data structure
    examples = defaultdict(list)

    # Generate a start state
    if env == "goto":
        env_id = FIXINSTGOTO_ENVS[0]
        env = gym.make(
            env_id,
            highlight=False,
            action_space=ActionSpace.all,
            num_dists=0,
            action_space_agent_color=False,
        )
    elif env == "goto_large":
        env_id = "BabyAI-GoToObjMazeS7-v0"
        env = gym.make(
            env_id,
            highlight=False,
            action_space=ActionSpace.all,
            num_dists=7,
            action_space_agent_color=False,
        )
    else:
        raise NotImplementedError(f"This environment has not been implemented")

    env = FullyObsWrapper(env)
    _ = env.reset()[0]
    grid = env.grid
    cleaned_grid = []
    for elem in grid.grid:
        if isinstance(elem, minigrid.core.world_object.Wall):
            cleaned_grid.append(elem)
        elif elem is None:
            cleaned_grid.append(elem)
        else:
            cleaned_grid.append(None)

    grid.grid = cleaned_grid
    env.set_state(grid, (3, 3), 0, 0)

    # All actions to test
    actions = list(range(4)) + list(range(7, 19))
    for action_space in ActionSpace:
        legal_actions = action_space.get_legal_actions()
        grid.grid = cleaned_grid
        env.set_state(grid, (3, 3), 3, 0)
        obs = env.step(6)
        state = obs[0]["image"]
        examples[action_space].append(state)

        if action_space == ActionSpace.all:
            continue

        for action in actions:
            grid.grid = cleaned_grid
            env.set_state(grid, (3, 3), 3, 0)
            if action not in legal_actions:
                action = 6
            obs = env.step(action)
            state = obs[0]["image"]
            examples[action_space].append(state)

    # Format the contexts
    for action_space, example in examples.items():
        examples[action_space] = [blosc.pack_array(np.stack(example, axis=0))]

    # Save examples
    filename = f"slot_per_action.pkl"
    with open(os.path.join(save_directory, filename), "wb") as file:
        pickle.dump(examples, file)


def generate_action_space_random(
    env_name: str,
    seed: int,
    num_distractors: int,
    n_examples: int,
    save_directory: str,
):
    """
    Generate a example demonstrations to condition on.

    Args:
        env_name: Name of the environment
        seed: Seed to use for the generation
        num_distractors: Number of distractors in the environment
        n_examples: Number of examples to generate
        save_directory: Directory to save the examples to

    """

    # Set up datastructure
    example_contexts = defaultdict(list)

    # Iterate over the action spaces
    for _ in range(n_examples):
        for action_space in range(8):
            action_space = ActionSpace(action_space)
            legal_actions = action_space.get_legal_actions()

            if env_name == "goto":
                # Sample an environment
                env_id = random.choice(FIXINSTGOTO_ENVS)
                env = gym.make(
                    env_id,
                    highlight=False,
                    action_space=action_space,
                    num_dists=num_distractors,
                )
                irrelevant_actions = GO_TO_IRRELEVANT_ACTIONS

            elif env_name == "goto_large":
                env_id = "BabyAI-GoToObjMazeS7-v0"
                env = gym.make(
                    env_id,
                    highlight=False,
                    action_space=action_space,
                    num_dists=num_distractors,
                )
                irrelevant_actions = GO_TO_IRRELEVANT_ACTIONS

            else:
                raise NotImplementedError(
                    f"This environment has not been implemented {env_name}"
                )
            env = FullyObsWrapper(env)

            success = False
            while not success:
                obs = env.reset()[0]
                images = [obs["image"]]
                executed_actions = []
                while len(executed_actions) < len(legal_actions):
                    # Get possible actions at current state

                    possible_actions = get_possible_actions(obs["image"])
                    legal_possible_actions = [
                        action
                        for action in legal_actions
                        if action in possible_actions and action not in executed_actions
                    ]

                    if len(legal_possible_actions) == 0:
                        break

                    for action in legal_possible_actions:
                        if action in irrelevant_actions:
                            executed_actions.append(action)
                            continue

                        new_obs, _, _, _, _ = env.step(action)
                        obs = new_obs
                        images.append(obs["image"])
                        executed_actions.append(action)

                if len(executed_actions) == len(legal_actions):
                    success = True
                    compressed_video = blosc.pack_array(np.stack(images, axis=0))
                    example_contexts[action_space].append(compressed_video)
                else:
                    seed += 1

    # Save example contexts
    filename = f"{env_name}_{n_examples}_{num_distractors}_action_space_random.pkl"
    with open(os.path.join(save_directory, filename), "wb") as file:
        pickle.dump(example_contexts, file)

    # Get longest episode
    for action_space, examples in example_contexts.items():
        longest_episode = max(
            [blosc.unpack_array(example).shape[0] for example in examples]
        )
        print(f"Action space {action_space} longest episode {longest_episode}")


def get_possible_actions(obs: np.array):
    """
    Get the possible actions at the current environment state

    Args:
        obs: Observation from the environment

    Returns:
        possible_actions: List of possible actions
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
    parser = argparse.ArgumentParser(
        description="Generate example trajectories for the planner"
    )
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    set_seed(config["seed"])
    register_envs()
    generate_example_trajectories(config)
