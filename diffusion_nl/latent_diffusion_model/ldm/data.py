import os
import pickle
import random
import time

import numpy as np

from einops import rearrange
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import tensorflow_datasets as tfds

import torch
from torchvision.transforms import Resize, InterpolationMode


from diffusion_nl.utils.data import (
    RLDSSpec,
    TrajectoryTransformBuilder,
    n_step_pattern_builder,
)


TextID2ID = {"taco_play": 0, "calvin": 1, "bridge": 2, "io_ai_tech": 3, "robonet": 4}
TextID2GoalType = {"taco_play": "both", "calvin": "both", "bridge": "both", "io_ai_tech": "image", "robonet": 4}

class RoboNetDataset(Dataset):
    pass


class CalvinImageDataset(Dataset):

    def __init__(
        self,
        datapath,
        mean,
        std,
        width,
        height,
        episodes,
        length,
        n_frames,
        examples,
    ):

        self.idd = "calvin"
        self.datapath = datapath
        self.episodes = episodes
        self.n_frames = n_frames
        self.examples = examples
        self.width = width
        self.height = height
        self.mean = mean
        self.std = std
        self.resize = Resize(
            size=(self.width, self.height), interpolation=InterpolationMode.BILINEAR
        )
        self.length = length
        self.n_episodes = self.episodes.shape[0]

    def __getitem__(self, index):
        # Sample a trajectory
        index = random.randint(0, self.n_episodes - 1)
        episode_start, episode_end = self.episodes[index]
        start = random.randint(episode_start, episode_end - self.n_frames)
        end = start + self.n_frames

        # Get the video
        video = self.get_video(start, end)
        video = (video - self.mean) / self.std
        video = rearrange(video, "t h w c -> t c h w")
        video = self.resize(video)
        video = rearrange(video, "t c h w -> t h w c")

        # Get the goal
        goal = video[-1]
        goal_type = 0

        # Get the example
        example = torch.from_numpy(random.choice(self.examples[self.idd])[0]).float()
        example = (example - self.mean) / self.std
        example = rearrange(example, "t h w c -> t c h w")
        example = self.resize(example)
        example = rearrange(example, "t c h w -> t h w c")

        return video, example, goal, goal_type, TextID2ID[self.idd]

    def get_video(self, start, end):
        video = []
        for idx in range(start, end):
            idx = "0" * (7 - len(str(idx))) + str(idx)
            filepath = os.path.join(self.datapath, f"episode_{idx}.npz")
            data = np.load(filepath, allow_pickle=True)
            for key, value in data.items():
                if key == "rgb_static":
                    video.append(value)
                    break

        video = torch.from_numpy(np.stack(video, axis=0)).float()
        return video

    def __len__(self):
        return self.length


class CalvinLangDataset(Dataset):

    def __init__(
        self,
        datapath,
        mean,
        std,
        width,
        height,
        episodes,
        n_frames,
        instruction2embeddings,
        examples,
    ):
        self.idd = "calvin"
        self.datapath = datapath
        self.episodes = episodes
        self.n_frames = n_frames
        self.instruction2embeddings = instruction2embeddings
        self.examples = examples
        self.width = width
        self.height = height
        self.mean = mean
        self.std = std
        self.resize = Resize(
            size=(self.width, self.height), interpolation=InterpolationMode.BILINEAR
        )

    def __getitem__(self, index):
        
        # Sample a trajectory
        episode_start, episode_end, instruction = self.episodes[index]
        start = random.randint(episode_start, episode_end - self.n_frames)
        end = start + self.n_frames

        # Get the video
        video = self.get_video(start, end)
        video = (video - self.mean) / self.std
        video = rearrange(video, "t h w c -> t c h w")
        video = self.resize(video)
        video = rearrange(video, "t c h w -> t h w c")

        # Get the goal
        goal = self.instruction2embeddings[instruction]
        goal_type = 1

        # Get the example
        example = torch.from_numpy(random.choice(self.examples[self.idd])[0]).float()
        example = (example - self.mean) / self.std
        example = rearrange(example, "t h w c -> t c h w")
        example = self.resize(example)
        example = rearrange(example, "t c h w -> t h w c")

        return video, example, goal, goal_type, TextID2ID[self.idd]

    def get_video(self, start, end):
        video = []
        for idx in range(start, end):
            idx = "0" * (7 - len(str(idx))) + str(idx)
            filepath = os.path.join(self.datapath, f"episode_{idx}.npz")
            data = np.load(filepath, allow_pickle=True)
            for key, value in data.items():
                if key == "rgb_static":
                    video.append(value)
                    break
        video = torch.from_numpy(np.stack(video, axis=0)).float()
        return video

    def __len__(self):
        return len(self.episodes)


class OpenEDataset(Dataset):
    def __init__(
        self,
        dataset,
        width,
        height,
        mean,
        std,
        idd,
        n_steps,
        n_trajectories,
        instruction_prob,
        instruction2emb,
        examples,
    ):
        self.dataset = (
            dataset.shuffle(buffer_size=10)
            .repeat()
            .batch(1)
            .prefetch(buffer_size=3)
            .as_numpy_iterator()
        )
        self.idd = idd
        self.n_steps = n_steps
        self.n_trajectories = n_trajectories
        self.instruction_prob = instruction_prob  # Probability to condition on an instruction instead of goal image
        self.instruction2emb = instruction2emb
        self.width = width
        self.height = height
        self.mean = mean
        self.std = std
        self.resize = Resize(
            size=(self.width, self.height), interpolation=InterpolationMode.BILINEAR
        )
        self.examples = examples

    def __len__(self):
        return self.n_trajectories

    def __getitem__(self, index):
        sample = self.dataset.next()
        transformed_sample = self.transform(sample)
        return transformed_sample

    def transform(self, sample):
        # Transform Video
        video = torch.from_numpy(sample["observation"][0]).float()
        video = (video - self.mean) / self.std
        video = rearrange(video, "t h w c -> t c h w")
        video = self.resize(video)
        video = rearrange(video, "t c h w -> t h w c")

        # Transform Goal
        if random.random() > self.instruction_prob or TextID2GoalType[self.idd] == "image":
            # Goal as Image
            goal = video[-1]
            goal_type = 0
        else:
            # Goal as instruction
            goal = self.instruction2emb[sample["instruction"][0][0].decode("utf-8")]
            goal_type = 1

        # Transform Example
        example = torch.from_numpy(random.choice(self.examples[self.idd])[0]).float()
        example = (example - self.mean) / self.std
        example = rearrange(example, "t h w c -> t c h w")
        example = self.resize(example)
        example = rearrange(example, "t c h w -> t h w c")

        return video, example, goal, goal_type, TextID2ID[self.idd]


def collate_generator(mean, std):
    def collate(data):
        videos = []
        lengths = []
        goals = []
        goal_types = []
        examples = []
        agent_ids = []

        for sample in data:
            video, example_video, goal, goal_type, idd = sample
            videos.append(video)
            goals.append(goal)
            goal_types.append(goal_type)
            agent_ids.append(idd)
            lengths.append(video.shape[0])
            examples.append(example_video)

        lengths = torch.tensor(lengths, dtype=torch.long)
        videos = torch.stack(videos, dim=0)
        agent_ids = torch.tensor(agent_ids, dtype=torch.long)
        examples = torch.stack(examples, dim=0)

        return videos, lengths, examples, goals, goal_types, agent_ids

    return collate


def get_data_multi(config):

    # Create Dataset
    training_datasets = []
    validation_datasets = []

    # Load instruction2embeddings
    instruction2embeddings = torch.load(config["embeddings"]["path"])

    # Load examples
    with open(config["examples_path"], "rb") as file:
        dataset2examples = pickle.load(file)
    
    

    for dataset in config["datasets"]:
        if dataset == "calvin":

            if config["instruction_prob"] < 1:
                # Add Calvin Image Dataset

                training_episodes = np.load(
                    os.path.join(
                        config["datapaths"][dataset], "training", "ep_start_end_ids.npy"
                    )
                )
               

                training_dataset = CalvinImageDataset(
                    os.path.join(config["datapaths"][dataset], "training"),
                    config["mean"],
                    config["std"],
                    config["width"],
                    config["height"],
                    training_episodes,
                    config["n_trajectories"][dataset]["train"],
                    config["n_frames"],
                    dataset2examples,
                )

                training_datasets.append(training_dataset)

                validation_episodes = np.load(
                    os.path.join(
                        config["datapaths"][dataset],
                        "validation",
                        "ep_start_end_ids.npy",
                    )
                )
                validation_dataset = CalvinImageDataset(
                    os.path.join(config["datapaths"][dataset], "validation"),
                    config["mean"],
                    config["std"],
                    config["width"],
                    config["height"],
                    validation_episodes,
                    config["n_trajectories"][dataset]["test"],
                    config["n_frames"],
                    dataset2examples,
                )
                validation_datasets.append(validation_dataset)

            # Add Calvin Lang Dataset
            episode_file = os.path.join(
                config["datapaths"][dataset],
                "training",
                "lang_annotations",
                "auto_lang_ann.npy",
            )
            training_annotations = np.load(episode_file, allow_pickle=True).item()
            count = 0
            training_episodes = {}
            for instruction, (episode_start, episode_end) in zip(
                training_annotations["language"]["ann"],
                training_annotations["info"]["indx"],
            ):
                training_episodes[count] = (episode_start, episode_end, instruction)
                count += 1

            training_dataset = CalvinLangDataset(
                os.path.join(config["datapaths"][dataset], "training"),
                config["mean"],
                config["std"],
                config["width"],
                config["height"],
                training_episodes,
                config["n_frames"],
                instruction2embeddings,
                dataset2examples,
            )
            training_datasets.append(training_dataset)

            episode_file = os.path.join(
                config["datapaths"][dataset],
                "validation",
                "lang_annotations",
                "auto_lang_ann.npy",
            )
            validation_annotations = np.load(episode_file, allow_pickle=True).item()
            count = 0
            validation_episodes = {}
            for instruction, (episode_start, episode_end) in zip(
                validation_annotations["language"]["ann"],
                validation_annotations["info"]["indx"],
            ):
                validation_episodes[count] = (episode_start, episode_end, instruction)
                count += 1

            validation_dataset = CalvinLangDataset(
                os.path.join(config["datapaths"][dataset], "validation"),
                config["mean"],
                config["std"],
                config["width"],
                config["height"],
                validation_episodes,
                config["n_frames"],
                instruction2embeddings,
                dataset2examples,
            )
            validation_datasets.append(validation_dataset)

        elif dataset in ["taco_play", "bridge","io_ai_tech"]:
            b = tfds.builder_from_directory(builder_dir=config["datapaths"][dataset])
            splits = list(b.info.splits.keys())
            step_map = step_map_generator(
                config["observation_keys"][dataset], config["instruction_keys"][dataset]
            )
            rlds_spec = RLDSSpec(
                observation_info=b.info.features["steps"]["observation"],
                action_info=b.info.features["steps"]["action"],
            )
            if "train" in splits:
                ds = b.as_dataset(split="train")
                trajectory_transform = TrajectoryTransformBuilder(
                    rlds_spec,
                    step_map_fn=step_map,
                    pattern_fn=n_step_pattern_builder(config["n_frames"]),
                ).build(validate_expected_tensor_spec=False)
                trajectory_dataset = (
                    trajectory_transform.transform_episodic_rlds_dataset(ds)
                )

                training_dataset = OpenEDataset(
                    trajectory_dataset,
                    config["width"],
                    config["height"],
                    config["mean"],
                    config["std"],
                    dataset,
                    config["n_frames"],
                    config["n_trajectories"][dataset]["train"],
                    config["instruction_prob"],
                    instruction2embeddings,
                    dataset2examples,
                )
                training_datasets.append(training_dataset)

            if "test" in splits:
                ds = b.as_dataset(split="test")
                trajectory_transform = TrajectoryTransformBuilder(
                    rlds_spec,
                    step_map_fn=step_map,
                    pattern_fn=n_step_pattern_builder(config["n_frames"]),
                ).build(validate_expected_tensor_spec=False)
                trajectory_dataset = (
                    trajectory_transform.transform_episodic_rlds_dataset(ds)
                )
                validation_dataset = OpenEDataset(
                    trajectory_dataset,
                    config["width"],
                    config["height"],
                    config["mean"],
                    config["std"],
                    "taco_play",
                    config["n_frames"],
                    config["n_trajectories"][dataset]["test"],
                    config["instruction_prob"],
                    instruction2embeddings,
                    dataset2examples,
                )
                validation_datasets.append(validation_dataset)

    # Concatenate the datasets
    training_dataset = ConcatDataset(training_datasets)
    validation_dataset = ConcatDataset(validation_datasets)

    # Create Dataloaders
    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=config["batch_size"],
        num_workers = 10,
        shuffle=True,
        collate_fn=collate_generator(config["mean"], config["std"]),
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=config["batch_size"],
        num_workers = 10,
        shuffle=False,
        collate_fn=collate_generator(config["mean"], config["std"]),
    )

    return (
        training_dataloader,
        validation_dataloader,
        instruction2embeddings,
        dataset2examples,
    )


def get_data(config):
    if config["data"]["dataset"] == "multi":
        return get_data_multi(config["data"])
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")


def step_map_generator(observation_key, instruction_key):
    def step_map_fn(step):
        transformed_step = {}
        transformed_step["observation"] = step["observation"][observation_key]
        if instruction_key in step["observation"]:
            transformed_step["instruction"] = step["observation"][instruction_key]
        transformed_step["is_first"] = step["is_first"]
        transformed_step["is_last"] = step["is_last"]
        transformed_step["is_terminal"] = step["is_terminal"]

        return transformed_step

    return step_map_fn
