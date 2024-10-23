import argparse
import json
import time

import tensorflow as tf
import torch
import yaml

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from torchvision.transforms import InterpolationMode, Resize
from diffusion_nl.utils.data import (
    RLDSSpec,
    TrajectoryTransformBuilder,
    n_step_pattern_builder,
)

###################### Functions ######################


def episode2steps(episode):
    return episode["steps"]


def step_map_generator(observation_key):
    def step_map_fn(step):
        transformed_step = {}
        transformed_step["observation"] = step["observation"][
            observation_key
        ]  # Resize to be compatible with robo_net trajectory
        if "natural_language_instruction" in step["observation"]:
            transformed_step["instruction"] = step["observation"][
                "natural_language_instruction"
            ]
        transformed_step["is_first"] = step["is_first"]
        transformed_step["is_last"] = step["is_last"]
        transformed_step["is_terminal"] = step["is_terminal"]

        return transformed_step

    return step_map_fn


def get_info(config):
    info = {}

    # Get dataset
    b = tfds.builder_from_directory(
        builder_dir=f"../../../data/{config['dataset']}/{config['version']}"
    )

    # Observation Keys
    observation_keys = list(b.info.features["steps"]["observation"].keys())
    info["observation_keys"] = observation_keys
    print(observation_keys)

    mt_opt_rlds_spec = RLDSSpec(
        observation_info=b.info.features["steps"]["observation"],
        action_info=b.info.features["steps"]["action"],
    )

    trajectory_transform = TrajectoryTransformBuilder(
        mt_opt_rlds_spec,
        step_map_fn=step_map_generator(config["observation_key"]),
        pattern_fn=n_step_pattern_builder(config["trajectory_length"]),
    ).build(validate_expected_tensor_spec=False)

    # Splits
    splits = list(b.info.splits.keys())
    info["splits"] = splits

    # Number of Trajectories
    for split in splits:
        ds = b.as_dataset(split=split)
        trajectory_dataset = trajectory_transform.transform_episodic_rlds_dataset(ds)

        count = 0
        for elem in trajectory_dataset.batch(1).as_numpy_iterator():
            print(count)
            count += 1

        info[split] = count

    print(info)
    with open(f"./{config['dataset']}.json", "w") as file:
        json.dump(info, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "rb") as file:
        config = yaml.safe_load(file)

    get_info(config)
