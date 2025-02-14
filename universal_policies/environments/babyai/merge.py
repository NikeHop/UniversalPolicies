"""
Merge single agent datasets of the in-distribution agents into a mixed dataset.
"""

import argparse
import logging
import os
import pickle

import yaml


def merge(data_dir, n_episodes, use_agent_type, num_distractors, minimum_length):
    """
    Merge multiple datasets into a single dataset.

    Args:
        data_dir (str): The directory where the datasets are located.
        n_episodes (int): The number of episodes in each dataset.
        use_agent_type (bool): Whether agents are coloured by type.
        num_distractors (int): The number of distractors in each episode.
        minimum_length (int): The minimum length of each episode.

    Returns:
        None
    """

    in_dist_action_spaces = [
        "standard",
        "no_left",
        "no_right",
        "diagonal",
        "wsad",
        "dir8",
    ]
    all_files = []

    for action_space in in_dist_action_spaces:
        folder_name = f"{action_space}_{n_episodes}_{minimum_length}_{num_distractors}_{use_agent_type}_demos"
        filename = f"dataset_{n_episodes}.pkl"
        filepath = os.path.join(data_dir, folder_name, filename)
        all_files.append(filepath)

    logging.info(f"Merging the following files: {all_files}")

    samples = []
    for file in all_files:
        with open(file, "rb") as f:
            data = pickle.load(f)
            samples += data

    new_folder = (
        f"mixed_{n_episodes}_{minimum_length}_{num_distractors}_{use_agent_type}_demos"
    )
    new_folder_path = os.path.join(data_dir, new_folder)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    filepath = os.path.join(new_folder_path, f"dataset_{n_episodes}.pkl")
    with open(filepath, "wb+") as file:
        pickle.dump(samples, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="The directory where the datasets are located",
    )
    parser.add_argument(
        "--n_episodes",
        default=0,
        type=int,
        help="Number of episodes in the single dataset dataset",
    )
    parser.add_argument(
        "--use_agent_type",
        action="store_true",
        help="Whether the agent is coloured by type",
    )
    parser.add_argument(
        "--num_distractors",
        default=0,
        type=int,
        help="Number of distractors in the environment",
    )
    parser.add_argument(
        "--minimum_length",
        default=4,
        type=int,
        help="Minimum length of an episode",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_dir = os.path.join(config["data_directory"], config["env"])

    merge(
        data_dir,
        config["n_episodes"],
        config["use_agent_type"],
        config["num_distractors"],
        config["minimum_length"],
    )
