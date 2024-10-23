from collections import defaultdict
import os 
import pickle
import random 

import blosc
import gymnasium as gym
import matplotlib.pyplot as plt 
import numpy as np
import torch 
import torch.nn.functional as F 

from einops import rearrange
from minigrid.core.actions import ActionSpace
from minigrid.wrappers import RGBImgObsWrapper
from torchvision.io import write_video

from diffusion_nl.utils.environments import (
    seeding,
    ENVS,
    TEST_ENV,
    CLASS2ENV,
    FIXINSTGOTO_ENVS,
)

from diffusion_nl.diffusion_model.utils import transform_sample, transform_video

from diffusion_nl.diffusion_model.utils import state2img

def debug_check():
    path = "../data/baby_ai"
    filename = "no_padding_each_action.pkl"
    with open(os.path.join(path,filename),"rb") as file:
        data = pickle.load(file)[0]

    sample = 0
    lengths = defaultdict(int)

    for sample in data:
        arr = torch.from_numpy(blosc.unpack_array(sample[2]))
        lengths[arr.shape[0]] += 1


def dir_without_image_test():
    """
    Test whether agent orientation is correct. Direction + 1 should be at the same place as agent in the last channel.
    """
    path = "../data/baby_ai"
    filename = "standard_10000_4_demos_"
    for fname in os.listdir(path):
        if filename in fname:
            with open(os.path.join(path, fname), "rb") as file:
                data = pickle.load(file)[0]
                for sample in data:
                    arr = torch.from_numpy(blosc.unpack_array(sample[2]))
                    dir = sample[3]

                    indices = (*(np.where(arr[:-1] == 10)[:-1]), [2] * (arr.shape[0]-1))
                    assert torch.all(arr[indices] == (torch.tensor(dir) + 1)), f"Agent location in file {fname} does not match the obs"

def visualize(config):

    filepath = (
        "../../data/GOTO/no_right_83_4_0_True_demos/BabyAI-FixInstGoToOrangeBall-v0.pkl"
    )

    with open(filepath,"rb") as file:
        data = pickle.load(file)
        for sample in data:
            print(sample)
            arrays = blosc.unpack_array(sample[2])
            for k in range(arrays.shape[0]):
                print(arrays[k])
                image = state2img(arrays[k])
                plt.imshow(image)
                plt.savefig(f"./test_{k}.png")

            break

def check_interpolation(config):
    # Load image
    dataset = "../../data/baby_ai/standard_10_4_0_demos_FixInstGoToYellowKey.pkl"

    with open(dataset,"rb") as file:
        samples = pickle.load(file)[0]
    print(samples)
    for sample in samples:
        arrays = blosc.unpack_array(sample[2])
        for k,image in enumerate(arrays):
            print(image.shape)
            image = torch.tensor(image, dtype=torch.float).unsqueeze(0)
            image = rearrange(image,"b h w c -> b c h w")
            #image = F.interpolate(torch.tensor(image,dtype=torch.float),size=64,mode="nearest")
            image = rearrange(image.squeeze(0),"c h w -> h w c").numpy()
            plt.imshow(image)
            plt.savefig(f"./test_{k}.png")
            print(image.shape)

        break

    # Print image

    # Interpolate image

def test_img_env(config):
    action_space = ActionSpace(0)
    env_name = "BabyAI-GoToObj-v0"
    num_distractors = 0
    env = gym.make(env_name, highlight=False, action_space=action_space)
    env = RGBImgObsWrapper(env, tile_size=16)

    obs = env.reset()
    image = obs[0]["image"]

    plt.imshow(obs[0]["image"])
    plt.savefig("test.png")

    image = torch.tensor(image, dtype=torch.float).unsqueeze(0)
    image = rearrange(image,"b h w c -> b c h w")
    image = F.interpolate(torch.tensor(image,dtype=torch.float),size=64,mode="nearest")
    image = rearrange(image.squeeze(0),"c h w -> h w c").numpy()
    
    plt.imshow(image)
    plt.savefig("test_scaled.png")

def check_examples(config):

    datapath = "../../data/GOTO/no_padding_each_action_0.pkl"

    with open(datapath,"rb") as file:
        data = pickle.load(file)
    
    for key,examples in data.items():
        example = examples[0]
        video = blosc.unpack_array(example)
        print(video.shape)
        video = transform_video(video)
        print(video.shape)
        write_video(f"./test_example_{key}.mp4",video,fps=2)

def check_dataset():
    path = "../../../data/GOTO_LARGE/mixed_25000_4_7_False_demos/dataset_25000.pkl"
    with open(path,"rb") as file:
        data = pickle.load(file)
        print(len(data))
        print(data[0])

 

if __name__ == "__main__":

    config = {}
    check_dataset()
