"""
Visualizing example trajectories of the environment
"""
import argparse 
import pickle 
import random 

import blosc
import matplotlib.pyplot as plt
import numpy as np 
import torch 

from torchvision.io import write_video

from diffusion_nl.diffusion_model.utils import transform_video

def visualize(filepath,state=True):

    with open(filepath,"rb") as file:
        dataset = pickle.load(file)
        
    text, _, video, directions, _, _ , _ = random.choice(dataset)

    print(directions)
    video = blosc.unpack_array(video)
    if state:
        video = transform_video(video)
    video = torch.tensor(video)
    write_video(f"{text}.mp4",video,fps=3)
    

def visualize_examples(filepath):
    with open(filepath,"rb") as file:
        examples_dataset = pickle.load(file)
        
    for key, examples in examples_dataset.items():
        example = examples[10]
        video = np.stack(example,axis=0)
        print(video.shape)
        video = transform_video(video)
        video = torch.tensor(video)
        write_video(f"{key}.mp4",video,fps=3)
        


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="data/default.pkl")
    args = parser.parse_args()

    #visualize(args.filepath)

    visualize_examples(args.filepath)
    
    

