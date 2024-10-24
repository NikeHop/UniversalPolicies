import random
import os
import pickle

import blosc
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from transformers import AutoTokenizer, T5EncoderModel
from torch.utils.data import DataLoader, Dataset, random_split


class ImitationGoalObsDataset(Dataset):
    def __init__(self, data, max_gap=1):
        self.data = data
        self.max_gap = max_gap

    def __getitem__(self, idx):
        trajectory_info = self.data[idx]
        obss = blosc.unpack_array(trajectory_info[2])
        actions = trajectory_info[-3]
        agent_id = trajectory_info[-1]

        t = obss.shape[0]
        start_timestep = random.randint(0, t - 2)
        gap = random.randint(1, self.max_gap)
        goal_timestep = min(start_timestep + gap, t - 1)

        obs = obss[start_timestep]
        goal = obss[goal_timestep]
        action = actions[start_timestep]

        return obs, goal, action, agent_id

    def __len__(self):
        return len(self.data)


class ImitationInstructionDataset(Dataset):
    def __init__(self, data, inst2embeddings):
        self.data = data
        self.inst2embeddings = inst2embeddings

    def __getitem__(self, idx):
        trajectory_info = self.data[idx]
        instruction = trajectory_info[0]
        obss = blosc.unpack_array(trajectory_info[2])
        actions = trajectory_info[-3]
        agent_id = trajectory_info[-1]

        # Sample timestep
        t = obss.shape[0]
        timestep = random.randint(0, t - 2)

        obs = obss[timestep]
        action = actions[timestep]
        instruction = self.inst2embeddings[instruction]

        return obs, instruction, action, agent_id

    def __len__(self):
        return len(self.data)


def collate_goal(data):
    obss, goals, actions, agent_ids = [], [], [], []
    for obs, goal, action, agent_id in data:
        obs = torch.tensor(obs, dtype=torch.float).permute(2, 0, 1)
        obss.append(obs)
        goal = torch.tensor(goal, dtype=torch.float).permute(2, 0, 1)
        goals.append(goal)
        actions.append(action)
        agent_ids.append(agent_id)

    obss = torch.stack(obss, dim=0)
    goals = torch.stack(goals, dim=0)
    actions = torch.tensor(actions, dtype=torch.long)
    agent_ids = F.one_hot(torch.tensor(agent_ids, dtype=torch.long), num_classes=8)

    return obss, goals, actions, agent_ids


def collate_instruction(data):
    obss, instructions, actions, agent_ids = [], [], [], []
    for obs, instruction, action, agent_id in data:
        obs = torch.tensor(obs, dtype=torch.float).permute(2, 0, 1)
        obss.append(obs)
        instructions.append(instruction)
        actions.append(action)
        agent_ids.append(agent_id)

    obss = torch.stack(obss, dim=0)
    instructions = torch.stack(instructions, dim=0)
    actions = torch.tensor(actions, dtype=torch.long)
    agent_ids = torch.tensor(agent_ids, dtype=torch.long)

    return obss, instructions, actions, agent_ids


def get_data_goal_language_policy(config):
    # Load data
    with open(
        os.path.join(config["data"]["datapath"], config["data"]["filename"]), "rb"
    ) as file:
        data = pickle.load(file)

    # Create inst2embeddings
    inst2embeddings = get_embeddings(data, config)

    # Create dataset
    dataset = ImitationInstructionDataset(data, inst2embeddings)

    n = len(dataset)
    n_train = int(n * config["data"]["percentage"])
    n_test = n - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_instruction,
        batch_size=config["data"]["batch_size"],
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_instruction,
        batch_size=config["data"]["batch_size"],
        pin_memory=True,
    )

    return train_dataloader, test_dataloader


def get_data_goal_obs_policy(config):
    # Load data
    with open(
        os.path.join(config["data"]["directory"], config["data"]["filename"]), "rb"
    ) as file:
        data = pickle.load(file)

    # Create dataset
    dataset = ImitationGoalObsDataset(data, config["data"]["max_gap"])

    n = len(dataset)
    n_train = int(n * config["data"]["percentage"])
    n_test = n - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_goal,
        batch_size=config["data"]["batch_size"],
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_goal,
        batch_size=config["data"]["batch_size"],
        pin_memory=True,
    )

    return train_dataloader, test_dataloader


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


def get_data(config):
    if config["goal_type"] == "obs":
        return get_data_goal_obs_policy(config)
    elif config["goal_type"] == "language":
        return get_data_goal_language_policy(config)
    else:
        raise NotImplementedError("Goal type not implemented")
