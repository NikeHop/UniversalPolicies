"""
Preparing data for diffusion model training
"""

import os
import pickle
import random

from functools import partial

import blosc
import numpy as np
import torch
import torch.nn.functional as F
import tqdm as tqdm
import wandb

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split, Dataset

from diffusion_nl.utils.utils import get_embeddings
from minigrid.core.actions import ActionSpace, Actions


class BabyAIOfflineTrajDataset(Dataset):
    def __init__(
        self,
        data,
        n_frames,
        n_context_frames,
        step_frequency,
        inst2embed,
        example_path,
    ) -> None:
        super().__init__()
        self.data = data
        self.n_frames = n_frames
        self.n_example_frames = n_context_frames
        self.step_frequency = step_frequency
        self.inst2embed = inst2embed

        self.load_examples(example_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Select sample
        sample = self.data[index]

        # Get action space and corresponding example trajectory
        action_space = sample[-1]
        example = random.choice(self.examples[action_space])

        # Filter out potentially bad samples
        try:
            instruction = sample[0]
            instruction = self.inst2embed[instruction]
            video = blosc.unpack_array(sample[2])

            # Make sure all example videos have the same length
            example_video = blosc.unpack_array(example)
            n_padding_frames = self.n_example_frames - example_video.shape[0]
            example_video = np.concatenate(
                [np.zeros((n_padding_frames, *example_video.shape[1:])), example_video],
                axis=0,
            )

        except Exception as e:
            return None, None, None, None

        # Subsample video with the given step frequency
        n_frames = video.shape[0]
        subsamples_video = []
        for i in range(n_frames):
            if i == 0 or i == n_frames - 1:
                subsamples_video.append(video[i])
            elif i % self.step_frequency == 0:
                subsamples_video.append(video[i])
            else:
                continue
        video = np.stack(subsamples_video, axis=0)

        # Repeat the final frame self.n_frame times
        video = np.concatenate(
            (
                video,
                np.tile(np.expand_dims(video[-1], axis=0), (self.n_frames, 1, 1, 1)),
            ),
            axis=0,
        )
        start = torch.randint(0, n_frames - 1, (1,)).item()
        video = video[start : start + self.n_frames]

        # Get agent id
        agent_id = sample[-1]

        return video, example_video, instruction, agent_id

    def load_examples(self, example_path):
        with open(example_path, "rb") as file:
            self.examples = pickle.load(file)


# Collate function for BabyAI dataset
def collate_babyai(data, mean, std, context_type):
    tasks = []
    videos = []
    example_videos = []
    lengths = []
    agent_ids = []
    action_spaces = []

    for video, example_video, instruction, agent_id in data:
        if video is None:
            continue

        video = torch.tensor(video, dtype=torch.float)
        example_video = torch.tensor(example_video, dtype=torch.float)
        videos.append(video)
        example_videos.append(example_video)
        tasks.append(instruction.reshape(1, -1))
        lengths.append(video.shape[0])
        agent_ids.append(agent_id)
        action_space = ActionSpace(agent_id)
        legal_actions = [int(a) for a in action_space.get_legal_actions()]
        actions = torch.tensor(
            [1 if i in legal_actions else 0 for i in range(len(Actions))]
        ).float()
        action_spaces.append(actions)

    example_videos = pad_sequence(example_videos, batch_first=True)
    videos = pad_sequence(videos, batch_first=True)
    tasks = torch.cat(tasks, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    agent_ids = torch.tensor(agent_ids, dtype=torch.long)
    action_spaces = torch.stack(action_spaces, dim=0)

    # Preprocessing
    videos = (videos - mean) / std
    example_videos = (example_videos - mean) / std

    # Masking
    seq_mask = torch.arange(videos.shape[1])[None, :] < lengths[:, None]
    mask = torch.ones_like(videos)
    mask[~seq_mask] = 0
    mask = mask.bool()

    if context_type == "time" or context_type == "channel":
        context = example_videos

    elif context_type == "agent_id":
        context = agent_ids

    elif context_type == "action_space":
        context = action_spaces

    else:
        raise NotImplementedError(f"The context type {context_type} is not implemented")

    return videos, mask, context, tasks


# Get Dataloader functions
def get_data(config):
    if config["data"]["dataset"] == "BabyAI":
        return get_data_baby_ai(config)
    else:
        raise ValueError("Dataset not supported")


def get_data_baby_ai(config):
    # Create Dataset
    with open(config["data"]["data_path"], "rb") as file:
        data = pickle.load(file)

    # Create embeddings
    inst2embed = get_embeddings(data, config["data"])

    # Create Dataset
    dataset = BabyAIOfflineTrajDataset(
        data,
        config["data"]["n_frames"],
        config["data"]["n_context_frames"],
        config["data"]["step_frequency"],
        inst2embed,
        config["data"]["example_path"],
    )
    example_contexts = dataset.examples

    # Split into training and evaluation
    n = len(dataset)
    n_train = int(n * config["data"]["train_split"])
    n_test = n - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

    # Log training examples:
    model_directory = os.path.join(
        config["logging"]["model_directory"],
        config["logging"]["experiment_name"],
        config["logging"]["project"],
        wandb.run.id,
    )

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # Collating function
    collate_babyai_partial = partial(
        collate_babyai,
        mean=config["data"]["mean"],
        std=config["data"]["std"],
        context_type=config["model"]["error_model"]["context_conditioning_type"],
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config["data"]["batch_size"],
        pin_memory=True,
        collate_fn=collate_babyai_partial,
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=config["data"]["batch_size"],
        pin_memory=True,
        collate_fn=collate_babyai_partial,
    )

    return train_dataloader, test_dataloader, inst2embed, example_contexts
