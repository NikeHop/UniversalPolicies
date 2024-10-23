"""
Preparing data for diffusion model training
"""

import os
import random

from functools import partial
import pickle

import blosc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm as tqdm
import wandb

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split, Dataset
from transformers import AutoTokenizer, T5EncoderModel

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
        sample = self.data[index]
        action_space = sample[-1]
        example = random.choice(self.examples[action_space])
        try:
            instruction = sample[0]
            instruction = self.inst2embed[instruction]
            video = blosc.unpack_array(sample[2])
            example_video = blosc.unpack_array(example)
            # pad example video to the n_context_frames
            n_padding_frames = self.n_example_frames - example_video.shape[0]
            example_video = np.concatenate(
                [np.zeros((n_padding_frames, *example_video.shape[1:])), example_video],
                axis=0,
            )

        except Exception as e:
            print(e)
            return None, None, None, None

        # Subsample video
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

    def compute_labels(self):
        self.instructions = {}
        for sample in self.data:
            instruction = sample[0]
            if instruction not in self.instructions:
                self.instructions[instruction] = len(self.instructions)
        print(self.instructions)

    def load_examples(self, example_path):
        with open(example_path, "rb") as file:
            self.examples = pickle.load(file)

    
# Collate function for BabyAI dataset
def collate_babyai(data, mean, std, context_type):
    # Batching
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
        actions = torch.tensor([1 if i in legal_actions else 0 for i in range(len(Actions))]).float()
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

def collate(data):
    pass


# Get Dataloader functions
def get_data(config):
    if config["data"]["dataset"] == "BabyAI":
        return get_data_baby_ai(config)
    elif config["data"]["dataset"] == "multi":
        return get_data_multi(config)
    else:
        raise ValueError("Dataset not supported")

def get_data_multi(config):
    # Create Dataset 
    training_datasets = []
    validation_datasets = []

    for dataset in config["dataset_names"]:
        if dataset=="calvin":
            pass

    training_dataloader = DataLoader(dataset=training_datasets, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], collate_fn=collate)
    validation_dataloader = DataLoader(dataset=validation_datasets, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], collate_fn=collate)

    return training_dataloader, validation_dataloader


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
        context_type=config["model"]["error_model"][
            "context_conditioning_type"
        ],
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


def get_embeddings(data, config):
    instructions = list(set([sample[0] for sample in data]))

    if config["embeddings"]["type"] == "random":
        inst2embed = get_random_embeddings(instructions, config)

    elif config["embeddings"]["type"] == "t5":
        inst2embed = get_t5_embeddings(instructions, config)

    else:
        raise NotImplementedError("Embedding type is not implemented")

    return inst2embed


def get_random_embeddings(instructions, config):
    inst2embed = {}
    embeddings = nn.Embedding(len(instructions), config["embeddings"]["size"])
    embeddings.requires_grad = False
    for i, inst in enumerate(sorted(instructions)):
        inst2embed[inst] = embeddings(torch.tensor(i))

    return inst2embed


def get_t5_embeddings(instructions, config):
    inst2embed = {}
    tokenizer = AutoTokenizer.from_pretrained(config["embeddings"]["model"])
    encoder_model = T5EncoderModel.from_pretrained(config["embeddings"]["model"]).to(
        config["embeddings"]["device"]
    )
    inputs = tokenizer(
        instructions, return_tensors="pt", padding=True, truncation=True
    ).to(config["embeddings"]["device"])

    encoded_embeddings = []
    n_instructions = len(instructions)
    n_encoded_instructions = 0
    B = config["embeddings"]["batch_size"]
    pbar = tqdm.tqdm(total=n_instructions)
    with torch.no_grad():
        while n_encoded_instructions < n_instructions:
            model_output = encoder_model(
                input_ids=inputs["input_ids"][
                    n_encoded_instructions : n_encoded_instructions + B
                ],
                attention_mask=inputs["attention_mask"][
                    n_encoded_instructions : n_encoded_instructions + B
                ],
            )
            encoded_embeddings.append(
                model_output.last_hidden_state.mean(dim=1).detach().cpu()
            )
            n_encoded_instructions += B
            pbar.update(B)

    embeddings = torch.cat(encoded_embeddings, dim=0)

    for elem, instruction in zip(embeddings, instructions):
        inst2embed[instruction] = elem

    return inst2embed
