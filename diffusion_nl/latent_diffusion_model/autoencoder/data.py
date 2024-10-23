"""
Data for training AutoEncoder
"""

import os
import pickle
import random

from functools import partial


import blosc
import glob
import numpy as np
import tensorflow_datasets as tfds
import torch

from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision.transforms import Resize, InterpolationMode

DATASETID2IMAGEKEY = {"calvin":"rgb_static", "taco_play":"rgb_static","io_ai_tech":"image"}
DATASETID2MEAN = {"calvin":torch.tensor(127.5), "taco_play":torch.tensor(127.5),"io_ai_tech":torch.tensor(127.5)}
DATASETID2STD = {"calvin":torch.tensor(127.5), "taco_play":torch.tensor(127.5),"io_ai_tech":torch.tensor(127.5)}
    

class CalvinImageDataset(Dataset):
    def __init__(self, datapath, width, height):
        self.id = "calvin"
        self.datapath = datapath
        self.mean = torch.tensor(127.5)
        self.std = torch.tensor(127.5)
        self.width = width 
        self.height = height
        self.resize = Resize(size=(self.width,self.height),interpolation=InterpolationMode.BILINEAR)
        self.frames = get_calvin_frames(datapath)

    def __getitem__(self, index):
        frame = self.frames[index]
        img = self.load_frame(frame)
        return self.transform(img)

    def transform(self, img):
        img = torch.from_numpy(img).float()
        img = (img-self.mean)/self.std
        img = self.resize(img.permute(2,0,1)).permute(1,2,0)
        return img

    def load_frame(self, idx):
        idx = "0" * (7 - len(str(idx))) + str(idx)
        filepath = os.path.join(self.datapath, f"episode_{idx}.npz")
        data = np.load(filepath, allow_pickle=True)
        for key,value in data.items():
            if key=="rgb_static":
                return value

    def __len__(self):
        return len(self.frames)


class OpenEImageDataset(Dataset):
    def __init__(self, datapath, width, height):
        self.datapath = datapath
        self.files = glob.glob(os.path.join(self.datapath,"*.npz"))
        self.length = 10*len(self.files)
        print(f"Dataset Length: {self.length}")
        self.mean = torch.tensor(127.5)
        self.std = torch.tensor(127.5)
        self.width = width 
        self.height = height
        self.resize = Resize(size=(self.width,self.height),interpolation=InterpolationMode.BILINEAR)

    def __len__(self):
        return self.length
    
    def __getitem__(self,index):
        index = index
        sample = random.choice(self.files)
     
        if "robonet" in sample:
            imgs = np.load(sample)["arr_0"]
        else:    
            imgs = np.load(sample)["arr_0"][:,0]
        
        n_imgs = imgs.shape[0]
        i = random.randint(0,n_imgs-1)
        return self.transform(imgs[i])
        
    def transform(self,img):
        img = torch.from_numpy(img).float()
        img = (img-self.mean)/self.std
        img = self.resize(img.permute(2,0,1)).permute(1,2,0)
        return img 

class BabyAIImageDataset(Dataset):

    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        try:
            video = blosc.unpack_array(sample[2])
        except Exception as e:
            return None, None

        n_frames = video.shape[0]
        image_id = random.randint(0, n_frames - 1)
        image = video[image_id]

        return image


def collate_babyai(data, mean, std):
    data = [torch.tensor(d, dtype=torch.float) for d in data]
    images = torch.stack(data, dim=0)
    images = (images - mean) / std
    return images

def collate(data):
    imgs = []

    for img in data:
        imgs.append(img)

    imgs = torch.stack(imgs,dim=0)

    return imgs

def get_data(config):
    print(config["data"]["dataset"])
    if config["data"]["dataset"] == "babyai":
        return get_data_babyai(config)
    elif config["data"]["dataset"] == "multi":
        return get_data_multi(config)
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")


def get_data_multi(config):

    training_datasets = []
    validation_datasets = []
    for dataset_name in config["data"]["datasets"]:
        print(dataset_name)
        if dataset_name == "calvin":
            training_path = os.path.join(
                config["data"]["data_paths"][dataset_name], "training"
            )
            training_dataset = CalvinImageDataset(training_path,config["data"]["width"],config["data"]["height"])
            training_datasets.append(training_dataset)

            validation_path = os.path.join(
                config["data"]["data_paths"][dataset_name], "validation"
            )
            validation_dataset = CalvinImageDataset(validation_path,config["data"]["width"],config["data"]["height"])
            validation_datasets.append(validation_dataset)
        
        elif dataset_name == "open_e":
            print("We are adding the Open Embodiment Dataset")
            training_path = os.path.join(
                config["data"]["data_paths"][dataset_name], "training"
            )
            training_dataset = OpenEImageDataset(training_path,config["data"]["width"],config["data"]["height"])
            training_datasets.append(training_dataset)

            validation_path = os.path.join(
                config["data"]["data_paths"][dataset_name], "validation"
            )
            validation_dataset = OpenEImageDataset(validation_path,config["data"]["width"],config["data"]["height"])
            validation_datasets.append(validation_dataset)

    training_dataset = ConcatDataset(training_datasets)
    validation_dataset = ConcatDataset(validation_datasets)

    training_dataloader = DataLoader(
        dataset=training_dataset,
        collate_fn=collate,
        shuffle=True,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        collate_fn=collate,
        shuffle=False,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    return training_dataloader, validation_dataloader


def get_data_babyai(config):
    print("Inside babyai")
    with open(config["data"]["data_path"], "rb") as file:
        data = pickle.load(file)

    dataset = BabyAIImageDataset(data)

    # Split into training and evaluation
    n = len(dataset)
    n_train = int(n * config["data"]["train_split"])
    n_test = n - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

    collate_fn = partial(
        collate_babyai, mean=config["data"]["mean"], std=config["data"]["std"]
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config["data"]["batch_size"],
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=config["data"]["batch_size"],
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_dataloader, test_dataloader

def episode2steps(episode):
    return episode["steps"]

def get_calvin_frames(datapath):
    # Load episode dataset
    episode_filepath = os.path.join(datapath, "ep_start_end_ids.npy")
    episodes = np.load(episode_filepath, allow_pickle=True)

    # For each episode add frames
    frames = []
    for eps in episodes:
        start, end = eps
        frames.extend(list(range(start, end + 1)))

    return frames
