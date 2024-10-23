"""
Test the model on the training set
"""

import argparse
import os 
import pickle 

import blosc
import matplotlib.pyplot as plt
import numpy as np 
import torch 
import yaml 

from diffusion_nl.diffusion_model.model import StateSpaceDiffusionModel
from diffusion_nl.diffusion_model.train import get_model_class
from diffusion_nl.diffusion_model.utils import transform_sample, state2img, transform_video

def evaluate(config):
    # Load the model
    model_class = get_model_class(config["model_type"])
    model = model_class.load_from_checkpoint(config["checkpoint"])
    model.eval()

    # Load the dataset
    datapath = "/var/scratch/nrhopner/experiments/diffusion_nl/data/GOTO/mixed_83_4_0_True_demos/dataset_83.pkl"
    with open(datapath, "rb") as file:
        dataset = pickle.load(file)

    arr = blosc.unpack_array(dataset[0][2])
    obs_0 = torch.tensor(arr[0]).float().to(config["device"]).unsqueeze(0)
    label = torch.zeros(1,128).to(config["device"])
    context = torch.zeros(1,1,8,8,3).to(config["device"])

    obs_0, label, sample = model.conditional_sample(obs_0,context,label)
    
    video = sample[0].int().cpu().numpy()
    video = transform_video(video)

    count = 0
    for elem in video:
        print(elem.shape)
        plt.imshow(elem)
        plt.savefig(f"{count}.png")
        count += 1
    
    #video = transform_video(video)



if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, required=True)
    args = argparser.parse_args()

    
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    
    with torch.no_grad():
        evaluate(config)
    

   
