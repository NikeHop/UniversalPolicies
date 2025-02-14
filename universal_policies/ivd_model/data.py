import os
import pickle
import random

import blosc
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class BabyAI_IVD_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        sample = self.data[idx]
        video = blosc.unpack_array(sample[2])
        t = video.shape[0]
        timestep = random.randint(0, t - 2)
        obs1 = video[timestep]
        obs2 = video[timestep + 1]
        action = int(sample[-3][timestep])
        return obs1, obs2, action

    def __len__(self):
        return len(self.data)


def collate_babyai(data):
    obs1, obs2, actions = [], [], []
    for elem in data:
        obs1.append(torch.tensor(elem[0], dtype=torch.float).permute(2, 0, 1))
        obs2.append(torch.tensor(elem[1], dtype=torch.float).permute(2, 0, 1))
        actions.append(elem[2])

    obs1 = torch.stack(obs1, dim=0)
    obs2 = torch.stack(obs2, dim=0)
    actions = torch.tensor(actions, dtype=torch.long)

    return obs1, obs2, actions


def get_data_babyai(config):
    # Load data
    with open(os.path.join(config["datapath"]), "rb") as file:
        data = pickle.load(file)

    # Create dataset
    dataset = BabyAI_IVD_Dataset(data)

    n = len(dataset)
    n_train = int(n * config["percentage"])
    n_test = n - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=config["num_workers"],
        batch_size=config["batch_size"],
        pin_memory=True,
        collate_fn=collate_babyai,
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=True,
        num_workers=config["num_workers"],
        batch_size=config["batch_size"],
        pin_memory=True,
        collate_fn=collate_babyai,
    )

    return train_dataloader, test_dataloader


def get_data(config):
    if config["dataset"] == "babyai":
        return get_data_babyai(config)
    else:
        raise NotImplementedError(f"Unknown dataset {config['dataset']}")
