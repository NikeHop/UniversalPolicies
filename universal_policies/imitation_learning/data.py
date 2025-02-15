import random
import pickle

import blosc
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, random_split

from universal_policies.utils.utils import get_embeddings


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
    with open(config["data"]["datapath"], "rb") as file:
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
    with open(config["data"]["datapath"], "rb") as file:
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


def get_data(config):
    if config["goal_type"] == "obs":
        return get_data_goal_obs_policy(config)
    elif config["goal_type"] == "language":
        return get_data_goal_language_policy(config)
    else:
        raise NotImplementedError("Goal type not implemented")
