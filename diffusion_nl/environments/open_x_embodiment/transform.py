"""
Transform the Open-Embodiment datasets for the Autoencoder to use 
"""

import argparse
import logging
import json
import os
import pickle

import numpy as np
import torch
import yaml

import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

from transformers import AutoTokenizer, AutoModel

from diffusion_nl.diffusion_model.data import get_t5_embeddings
from diffusion_nl.utils.data import (
    RLDSSpec,
    TrajectoryTransformBuilder,
    n_step_pattern_builder,
)
from diffusion_nl.latent_diffusion_model.ldm.data import step_map_generator


DATASET2IMAGEKEY = {
    "io_ai_tech/0.0.1": "image",
    "taco_play/0.1.0": "rgb_static",
    "language_table/0.1.0": "rgb",
    "bridge/0.1.0": "image",
}


def transform_datasets_for_autoencoder(config):

    # Create directories to store data
    training_directory = os.path.join(config["data_directory"], "training")
    validation_directory = os.path.join(config["data_directory"], "validation")
    os.makedirs(training_directory, exist_ok=True)
    os.makedirs(validation_directory, exist_ok=True)
    logging.info("Directories created")

    # Convert taco-play dataset
    dataset_statistics = {}
    train_batch_count = 0
    valid_batch_count = 0
    for dataset in config["datasets"]:
        train_batch_count, valid_batch_count = transform_dataset(
            train_batch_count,
            valid_batch_count,
            training_directory,
            validation_directory,
            dataset,
        )
        dataset_statistics[dataset] = (train_batch_count, valid_batch_count)

        logging.info(f"{dataset} dataset transformed")

    with open("dataset_statistics.json", "w+") as file:
        json.dump(dataset_statistics, file)


def transform_robonet_dataset_autoencoder(config):
    pass

def transform_dataset(
    train_batch_count,
    valid_batch_count,
    training_directory,
    validation_directory,
    dataset,
):
    train_count = 0
    valid_count = 0
    image_key = DATASET2IMAGEKEY[dataset]

    # Load the dataset
    b = tfds.builder_from_directory(builder_dir=f"../../../data/{dataset}")
    splits = list(b.info.splits.keys())

    # Convert training dataset
    if "train" in splits:
        ds = b.as_dataset(split="train")
        ds = ds.map(episode2steps).flat_map(lambda x: x)

        batch = []
        for sample in tqdm.tqdm(ds.batch(1).as_numpy_iterator()):
            img = sample["observation"][image_key][0]
            batch.append(img)
            train_count += 1

            if train_count % 10 == 0:
                imgs = np.stack(batch, axis=0)
                np.savez(
                    os.path.join(training_directory, f"{train_batch_count}.npz"), imgs
                )
                batch = []
                train_batch_count += 1

        imgs = np.stack(batch, axis=0)
        np.savez(os.path.join(training_directory, f"{train_batch_count}.npz"), imgs)
        train_batch_count += 1

    # Convert the validation dataset
    if "test" in splits:
        ds = b.as_dataset(split="test")
        ds = ds.map(episode2steps).flat_map(lambda x: x)

        batch = []
        for sample in tqdm.tqdm(ds.batch(1).as_numpy_iterator()):
            img = sample["observation"][image_key][0]
            batch.append(img)
            valid_count += 1

            if valid_count % 10 == 0:
                imgs = np.stack(batch, axis=0)
                np.savez(
                    os.path.join(validation_directory, f"{valid_batch_count}.npz"), imgs
                )
                batch = []
                valid_batch_count += 1

        imgs = np.stack(batch, axis=0)
        np.savez(os.path.join(validation_directory, f"{valid_batch_count}.npz"), imgs)
        valid_batch_count += 1

    return train_batch_count, valid_batch_count


def episode2steps(episode):
    return episode["steps"]


def collect_instructions(config):
    # Collect all instructions from all datasets and obtain T5 embeddings
    instructions = []

    # CALVIN
    calvin_instructions = get_calvin_instructions(config)
    instructions += calvin_instructions
    print(f"Got CALVIN instructions: {len(calvin_instructions)}")

    # OPEN - Embodiment
    open_e_instructions = get_open_e_instructions(config)
    instructions += open_e_instructions
    print(f"Got Open Embodiment instructions: {len(open_e_instructions)}")

    # ROBONET
    robonet_instructions = get_robonet_instructions(config)
    instructions += robonet_instructions
    print(f"Got Robonet instructions {len(robonet_instructions)}")

    # Embed instructions
    instructions2embeddings = get_t5_embeddings(instructions, config)

    with open(
        os.path.join(config["savepath"], "instructions2embeddings.pt"), "wb+"
    ) as file:
        torch.save(instructions2embeddings, file)


def get_calvin_instructions(config):
    instructions = []
    for dataset in ["calvin_debug_dataset", "task_D_D"]:
        for split in ["training", "validation"]:
            filepath = os.path.join(
                config["datapaths"]["calvin"],
                dataset,
                split,
                "lang_annotations",
                "auto_lang_ann.npy",
            )
            annotations = np.load(filepath,allow_pickle=True).item()
            instructions += annotations["language"]["ann"]

    instructions = list(set(instructions))
    return instructions


def get_open_e_instructions(config):
    instructions = []

    for dataset in config["datasets"]:
        datapath = config["datapaths"][dataset]
        b = tfds.builder_from_directory(builder_dir=datapath)

        rlds_spec = RLDSSpec(
            observation_info=b.info.features["steps"]["observation"],
            action_info=b.info.features["steps"]["action"],
        )
        step_map = step_map_generator(
            config["observation_keys"][dataset], config["instruction_keys"][dataset]
        )

        trajectory_transform = TrajectoryTransformBuilder(
            rlds_spec,
            step_map_fn=step_map,
            pattern_fn=n_step_pattern_builder(config["n_frames"]),
        ).build(validate_expected_tensor_spec=False)

        splits = list(b.info.splits.keys())
        for split in splits:
            ds = b.as_dataset(split=split)
            trajectory_dataset = trajectory_transform.transform_episodic_rlds_dataset(
                ds
            )
            # Iterate over dataset and extract instructions
            for sample in trajectory_dataset.batch(1).as_numpy_iterator():
                instructions.append(sample["instruction"][0][0].decode("utf-8"))

    instructions = list(set(instructions))
    print("Number of instructions: ", len(instructions))

    return instructions


def get_robonet_instructions(config):
    # Load metadata file 
    return []


def get_examples_calvin(config):
    dataset2examples = {}
    examples = []

    with open(
        os.path.join(
            config["datapaths"]["calvin"],
            "task_D_D",
            "validation",
            "lang_annotations",
            "auto_lang_ann.npy",
        ),
        "rb",
    ) as file:
        annotations = np.load(file, allow_pickle=True).item()

    count = 0
    for inst, frames in zip(annotations["language"]["ann"], annotations["info"]["indx"]):
        start, end = frames 
        random_start = np.random.randint(start, end - config["n_frames"])
        random_end = random_start + config["n_frames"]
        video = get_video(os.path.join(config["datapaths"]["calvin"],"task_D_D","validation"),random_start, random_end)
        examples.append(video[None,:,:,:,:])
        count += 1
        if count>config["n_examples"]:
            break
    dataset2examples["calvin"] = examples

    return dataset2examples

def get_examples_open_embodiment(config):
    dataset2examples = {}
    for dataset in config["datasets"]:
        examples = []
        datapath = config["datapaths"][dataset]
        b = tfds.builder_from_directory(builder_dir=datapath)

        rlds_spec = RLDSSpec(
            observation_info=b.info.features["steps"]["observation"],
            action_info=b.info.features["steps"]["action"],
        )
        step_map = step_map_generator(
            config["observation_keys"][dataset], config["instruction_keys"][dataset]
        )

        trajectory_transform = TrajectoryTransformBuilder(
            rlds_spec,
            step_map_fn=step_map,
            pattern_fn=n_step_pattern_builder(config["n_frames"]),
        ).build(validate_expected_tensor_spec=False)

        ds = b.as_dataset(split="train")
        trajectory_dataset = trajectory_transform.transform_episodic_rlds_dataset(ds)

        # Iterate over dataset and extract instructions
        count = 0
        for sample in trajectory_dataset.batch(1).as_numpy_iterator():
            examples.append(sample["observation"])
            count += 1
            if count > config["n_examples"]:
                break

        dataset2examples[dataset] = examples
    
    return dataset2examples

def get_examples(config):
    dataset2examples = {}

    dataset2examples.update(get_examples_open_embodiment(config))
    dataset2examples.update(get_examples_calvin(config))

    with open(
        os.path.join(
            config["savepath"], f"multi_robot_{config['n_frames']}_{config['n_examples']}_examples.pkl"
        ),
        "wb+",
    ) as file:
        pickle.dump(dataset2examples, file)


def get_validation_trajectories_calvin(config):
    dataset2trajectories = {}
    trajectories = []

    with open(
        os.path.join(
            config["datapaths"]["calvin"],
            "task_D_D",
            "validation",
            "lang_annotations",
            "auto_lang_ann.npy",
        ),
        "rb",
    ) as file:
        annotations = np.load(file, allow_pickle=True).item()

    count = 0 
    for inst, frames in zip(annotations["language"]["ann"], annotations["info"]["indx"]):
        start, end = frames 
        random_start = np.random.randint(start, end - config["n_frames"])
        random_end = random_start + config["n_frames"]
        video = get_video(os.path.join(config["datapaths"]["calvin"],"task_D_D","validation"),random_start, random_end)
        trajectories.append((inst, video[None,:,:,:,:]))
        count += 1
        if count>config["n_trajectories"]:
            break
    
    dataset2trajectories["calvin"] = trajectories

    return dataset2trajectories

def get_video(datapath, start, end):
    video = []
    for idx in range(start, end):
        idx = "0" * (7 - len(str(idx))) + str(idx)
        filepath = os.path.join(datapath, f"episode_{idx}.npz")
        data = np.load(filepath, allow_pickle=True)
        for key, value in data.items():
            if key == "rgb_static":
                video.append(value)
                break

    video = np.stack(video, axis=0)
    return video

def get_validation_trajectories_open_embodiment(config):
    dataset2trajectories = {}
    for dataset in config["datasets"]:
        trajectories = []
        datapath = config["datapaths"][dataset]
        b = tfds.builder_from_directory(builder_dir=datapath)

        rlds_spec = RLDSSpec(
            observation_info=b.info.features["steps"]["observation"],
            action_info=b.info.features["steps"]["action"],
        )
        step_map = step_map_generator(
            config["observation_keys"][dataset], config["instruction_keys"][dataset]
        )

        trajectory_transform = TrajectoryTransformBuilder(
            rlds_spec,
            step_map_fn=step_map,
            pattern_fn=n_step_pattern_builder(config["n_frames"]),
        ).build(validate_expected_tensor_spec=False)

        ds = b.as_dataset(split="train")
        trajectory_dataset = trajectory_transform.transform_episodic_rlds_dataset(ds)

        # Iterate over dataset and extract instructions
        count = 0
        for sample in trajectory_dataset.batch(1).as_numpy_iterator():
            if "instruction" in sample:
                instruction = sample["instruction"][0][0].decode("utf-8")
            else:
                instruction = None 
            val_traj = (
                instruction,
                sample["observation"][0],
            )
            trajectories.append(val_traj)
            count += 1
            if count > config["n_trajectories"]:
                break

        dataset2trajectories[dataset] = trajectories
    
    return dataset2trajectories


def get_validation_trajectories(config):
    dataset2trajectories = {}
    
    open_e_dataset2trajectories = get_validation_trajectories_open_embodiment(config)
    dataset2trajectories.update(open_e_dataset2trajectories)
    calvin_dataset2trajectories = get_validation_trajectories_calvin(config)
    dataset2trajectories.update(calvin_dataset2trajectories)

    with open(
        os.path.join(
            config["savepath"],
            f"multi_robot_{config['n_frames']}_{config['n_trajectories']}_validation_trajectories.pkl",
        ),
        "wb+",
    ) as file:
        pickle.dump(dataset2trajectories, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./config.yaml", help="path to config file"
    )
    args = parser.parse_args()

    with open(args.config, "rb") as file:
        config = yaml.safe_load(file)

    logging.basicConfig(
        level=logging.INFO,  # Set the log level to DEBUG
        format="%(asctime)s - %(levelname)s - %(message)s",  # Define log message format
        datefmt="%Y-%m-%d %H:%M:%S",  # Optional: Customize the timestamp format
    )

    # transform_datasets_for_autoencoder(config)
    # collect_instructions(config)
    # get_examples(config)
    # get_validation_trajectories(config)

    transform_robonet_dataset_autoencoder(config)
