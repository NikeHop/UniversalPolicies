from cgi import test
import random
from time import time
import os
import pickle

import blosc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


class CALVIN_IVD_Dataset(Dataset):

    def __init__(self, datapath):
        self.datapath = datapath
        self.frames = get_calvin_frames_ivd(datapath)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        next_frame = frame + 1
        img, action = self.load_frame(frame)
        next_img, _ = self.load_frame(next_frame)
        return img, next_img, action

    def load_frame(self, idx):
        idx = "0" * (7 - len(str(idx))) + str(idx)
        filepath = os.path.join(self.datapath, f"episode_{idx}.npz")
        data = np.load(filepath, allow_pickle=True)
        for key, value in data.items():
            if key == "rgb_static":
                img = value
            if key == "rel_actions":
                action = value

        return img, action

    def __len__(self):
        return len(self.frames)


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


def collate_calvin(data):

    obs1, obs2, actions = [], [], []
    for elem in data:
        obs1.append(torch.tensor(elem[0], dtype=torch.float).permute(2, 0, 1))
        obs2.append(torch.tensor(elem[1], dtype=torch.float).permute(2, 0, 1))
        actions.append(torch.from_numpy(elem[2]).float())

    obs1 = torch.stack(obs1, dim=0)
    obs2 = torch.stack(obs2, dim=0)
    actions = torch.stack(actions, dim=0)

    return obs1, obs2, actions


def get_data_babyai(config):

    # Load data
    with open(os.path.join(config["path"], config["filename"]), "rb") as file:
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


def get_data_calvin(config):

    training_path = os.path.join(config["datapath"], "training")
    validation_path = os.path.join(config["datapath"], "validation")

    training_dataset = CALVIN_IVD_Dataset(training_path)
    validation_dataset = CALVIN_IVD_Dataset(validation_path)

    train_dataloader = DataLoader(
        training_dataset,
        shuffle=True,
        batch_size=config["batch_size"],
        pin_memory=True,
        collate_fn=collate_calvin,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=config["batch_size"],
        pin_memory=True,
        collate_fn=collate_calvin,
    )

    return train_dataloader, validation_dataloader


def get_data(config):
    if config["dataset"] == "babyai":
        return get_data_babyai(config)
    elif config["dataset"] == "calvin":
        return get_data_calvin(config)
    else:
        raise NotImplementedError(f"Unknown dataset {config['dataset']}")


def get_calvin_frames_ivd(datapath):
    # Load episode dataset
    episode_filepath = os.path.join(datapath, "ep_start_end_ids.npy")
    episodes = np.load(episode_filepath, allow_pickle=True)

    # For each episode add frames
    frames = []
    for eps in episodes:
        start, end = eps
        frames.extend(list(range(start, end)))

    return frames


if __name__ == "__main__":
    pass
