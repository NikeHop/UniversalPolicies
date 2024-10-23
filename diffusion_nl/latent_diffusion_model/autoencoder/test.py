"""
Visualize the reconstruction for each environment 
"""

import argparse 
import os 

import torch 
import yaml 

from einops import rearrange
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from diffusion_nl.latent_diffusion_model.autoencoder.data import OpenEImageDataset, collate, CalvinImageDataset
from diffusion_nl.latent_diffusion_model.autoencoder.model import Autoencoder

def check_latent_space_and_reconstruction(config):
    ae = Autoencoder.load_from_checkpoint(config["checkpoint"],device=config["device"])
    ae.eval()

    # Load datasets 
    dataset = CalvinImageDataset(os.path.join(config["datapath"],"validation"),128,128)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=collate)

    for imgs in dataloader:
        imgs = rearrange(imgs, "b h w c -> b c h w").to(config["device"])
        latent_imgs = ae.get_encoding(imgs)
        rec = ae.decode(latent_imgs)

        print(f"Max-Rec: {rec.max()}")
        print(f"Min-Rec: {rec.min()}")

        # Before standardization 
        latent_max = latent_imgs.max()
        latent_min = latent_imgs.min()
        range = latent_max-latent_min
        midpoint = range/2
        print(range, midpoint, latent_max, latent_min)
        print(f"Max: {latent_imgs.max()}")
        print(f"Min: {latent_imgs.min()}")
        
        add = latent_min+midpoint
        latent_imgs = (latent_imgs-add)/midpoint
        print(f"Max: {latent_imgs.max()}")
        print(f"Min: {latent_imgs.min()}")
        
        """
        mean = latent_imgs.mean(dim=(0,2,3))
        std = latent_imgs.std(dim=(0,2,3))
        print(std.shape)

        latent_imgs = rearrange(latent_imgs, "b c h w -> b h w c")
        latent_imgs = (latent_imgs/std)
        print(f"Max: {latent_imgs.max()}")
        print(f"Min: {latent_imgs.min()}")
        print(latent_imgs.std(dim=(0,1,2)))
        """
    
        # After standardization


def visualize_reconstruction(config):
    # Load Autoencoder 
    ae = Autoencoder.load_from_checkpoint(config["checkpoint"], device=config["device"])
    ae.eval()

    # Load datasets 
    dataset = OpenEImageDataset(config["datapath"],128,128)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=collate)

    count = 0 
    for imgs in dataloader:
        print(imgs.shape)
        print(type(imgs))
        imgs = rearrange(imgs, "b h w c -> b c h w").to(config["device"])
        encoded_imgs = ae.encode(imgs)
        print(encoded_imgs.shape)
        reconstruction, _ = ae(imgs)
        reconstruction = reconstruction*127.5 + 127.5
        imgs = imgs*127.5 + 127.5
       
        plt.clf()
        plt.imshow(imgs[0].cpu().permute(1,2,0).int())
        plt.savefig(f"original_{count}.png")
        plt.clf()
        plt.imshow(reconstruction[0].cpu().permute(1,2,0).int())
        plt.savefig(f"reconstruction_{count}.png")
        count += 1

        if count>config["n_samples"]:
            break
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    # Load configuration 
    with open(args.config,"rb") as file:
<<<<<<< HEAD
        config = yaml.load(open(args.config, "r"))

    visualize_reconstruction(config)
    
=======
        config = yaml.safe_load(open(args.config, "r"))

    with torch.no_grad():
        visualize_reconstruction(config)
        #check_latent_space_and_reconstruction(config)
>>>>>>> a93d8e4fbdbcf671bd564734bee51ff8eb41588e
