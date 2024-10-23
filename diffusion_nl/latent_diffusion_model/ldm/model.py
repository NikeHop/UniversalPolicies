import os 
import random 

import blosc 
import matplotlib.pyplot as plt
import numpy as np 
import torch
import wandb 

from diffusion_nl.utils.models import BaseDiffusion
from diffusion_nl.latent_diffusion_model.ldm.data import TextID2ID, TextID2GoalType
from diffusion_nl.latent_diffusion_model.autoencoder.model import Autoencoder
from diffusion_nl.utils.networks import GoalEncoder
from diffusion_nl.utils.networks import ConditionalUnet3D

from einops import rearrange

from torch.optim import Adam
from torchvision.transforms import Resize, InterpolationMode

class LatentDiffusionModel(BaseDiffusion):

    def __init__(self, env, validation_trajectories, config: dict = {}):
        self.debug = config["debug"]
        self.image_directory = config["image_directory"]
        self.sampling_strategy = config["sampling_strategy"]
        self.sampling_timesteps = list(range(1,config["T"]))
        

        # Eval config
        self.n_samples = config["eval"]["n_samples"]
        self.num_frames = config["eval"]["n_frames"]
        self.context_frames = config["eval"]["n_context_frames"]
        self.context_type = config["error_model"]["UNet"]["context_conditioning_type"]

        # Data Config
        self.image_size = config["image_size"]
        self.image_channel = config["image_channel"]

        model = load_error_model(config["error_model"])
        
        super().__init__(
            model,
            config["T"],
            config["variance_schedule"],
            config["beta_start"],
            config["beta_end"],
            config["lr"],
            config["mean"],
            config["std"],
            config["l1"],
            config["use_instruction"],
            config["use_context"],
            config["conditional_prob"],
            config["cond_w"],
            config["model_type"]
        )

        self.goal_encoder = GoalEncoder(config["error_model"]["UNet"]["time_channels"])
        self.validation_trajectories = validation_trajectories # Dictionary mapping dataset of tuple: (instruction, gt_video)
        self.ae = self.load_autoencoder(config)
        self.ae.freeze()
        self.resize = Resize(size=(config["width"],config["height"]),interpolation=InterpolationMode.BILINEAR)

        self.save_hyperparameters()

    def ae_encode(self,data):
        B = data.shape[0]
        data = rearrange(data,"b t h w c -> (b t) c h w")
        data = self.ae.get_encoding(data)
        data = rearrange(data,"(b t) c h w-> b t h w c",b=B)
        self.latent_min = data.flatten(start_dim=1,end_dim=4).min(dim=-1)[0]
        self.latent_max = data.flatten(start_dim=1,end_dim=4).max(dim=-1)[0]
        self.latent_midpoint = self.latent_min + (self.latent_max-self.latent_min)/2
        self.latent_factor = self.latent_max - self.latent_midpoint
        
        data = (data-self.latent_midpoint.reshape(-1,1,1,1,1))/self.latent_factor.reshape(-1,1,1,1,1)

        print(f"Min: {data.min()}, Max: {data.max()}")    
        
        return data
    
    def training_step(self, batch, batch_idx):
        # Unpack batch
        x0, lengths, examples, goals, goal_types, agent_ids = batch
        B = x0.shape[0]

        # Encode the trajectory
        if batch_idx==0:
            x0 = self.ae_encode(x0)
        else:
            x0 = self.ae_encode(x0)

        # Create the context
        if self.context_type == "agent_id":
            context = agent_ids
        elif self.context_type == "time":
            context = self.ae_encode(examples)
        else:
            raise ValueError(f"The context type {self.context_type} is not implemented")

        # Encode the goal images
        encoded_images = []
        for goal, goal_type in zip(goals, goal_types):
            if goal_type == 0:     
                goal = goal.unsqueeze(0)
                goal = rearrange(goal,"b h w c -> b c h w")
                goal = self.goal_encoder(goal).squeeze(0)
                encoded_images.append(goal)
       
            elif goal_type == 1:
            
                encoded_images.append(goal)
            else:
                raise ValueError(f"The goal type {goal_type} is not implemented")

        labels = torch.stack(encoded_images, dim=0)

        # Create Mask 
        seq_mask = torch.arange(x0.shape[1])[None, :].to(lengths.device) < lengths[:, None]
        masks = torch.ones_like(x0)
        masks[~seq_mask] = 0
        masks = masks.bool()

        # Send latent videos to diffusion model
        loss = super().training_step(x0, masks, context, labels, B)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        x0, lengths, examples, goals, goal_types, agent_ids = batch
        B = x0.shape[0]

        # Encode the trajectory
        if batch_idx==0:
            x0 = self.ae_encode(x0)
        else:
            x0 = self.ae_encode(x0)

        # Create the context
        if self.context_type == "agent_id":
            context = agent_ids
        elif self.context_type == "time":
            context = self.ae_encode(examples)
        else:
            raise ValueError(f"The context type {self.context_type} is not implemented")

        # Encode the goal images
        encoded_images = []
        for goal, goal_type in zip(goals, goal_types):
            if goal_type == 0:
                goal = goal.unsqueeze(0)
                goal = rearrange(goal,"b h w c -> b c h w")
                goal = self.goal_encoder(goal).squeeze(0)
                encoded_images.append(goal)
                
            elif goal_type == 1:
                encoded_images.append(goal)
            else:
                raise ValueError(f"The goal type {goal_type} is not implemented")

        labels = torch.stack(encoded_images, dim=0)

        # Create Mask 
        seq_mask = torch.arange(x0.shape[1])[None, :].to(lengths.device) < lengths[:, None]
        masks = torch.ones_like(x0)
        masks[~seq_mask] = 0
        masks = masks.bool()

        # Send latent videos to diffusion model
        print(x0.shape)
        super().validation_step(x0, masks, context, labels,B)
        
    def create_conditional_samples(self, n_samples):
        gt_videos = []
        sampled_videos = []
        instructions = []
        
        datasets = list(self.validation_trajectories.keys())
        
        val_datasets = []
        for key in datasets:
            if TextID2GoalType[key]=="both":
                val_datasets.append(key)
        
        for i in range(n_samples):
            dataset = random.choice(val_datasets)
            val_traj = random.choice(self.validation_trajectories[dataset])
            instruction, gt_video = val_traj
            gt_videos.append(gt_video[0])
            instructions.append(instruction)    
            obs_0 = torch.from_numpy(gt_video[0][0]).float().unsqueeze(0)
            label = self.instruction2embed[instruction]
            
            if self.context_type == "agent_id":
                k = TextID2ID[dataset]
                context = torch.tensor(k, dtype=torch.long).unsqueeze(0)

            elif self.context_type == "time" or self.context_type == "channel":
                context = random.choice(self.example_context[dataset])
                context = torch.from_numpy(context).float()
            else:
                raise NotImplementedError(f"Context type {self.context_type} not implemented")

            sampled_video = self.conditional_sample(
                obs_0, context, label,
            )
            sampled_videos.append(sampled_video)

        
        return gt_videos, sampled_videos, instructions
    
    def on_validation_epoch_end(self):
        gt_videos, sampled_videos, instructions = self.create_conditional_samples(self.n_samples)

        for gt_video, sampled_video, instruction in zip(gt_videos, sampled_videos, instructions):
            gt_video = rearrange(gt_video,"t h w c -> t c h w")
            sampled_video = rearrange(sampled_video.squeeze(0),"t h w c -> t c h w")
            sampled_video = (sampled_video * self.std + self.mean).astype(int)
            wandb.log({f"samples/{instruction}/gt_video": wandb.Video(gt_video, fps=1, format="gif")})
            wandb.log({f"samples/{instruction}/sampled_video": wandb.Video(sampled_video, fps=1, format="gif")})


    def conditional_sample(self, obs_0, context, label):
        # Normalize Data
        normalized_obs_0 = (obs_0 - self.mean) / self.std
        if self.context_type=="time" or self.context_type=="channel":
            context = (context - self.mean) / self.std

        # Send to inputs to device 
        ae_device = self.get_ae_device()

        # Encode via Autoencoder
        normalized_obs_0 = normalized_obs_0.to(ae_device)
        normalized_obs_0 = rearrange(normalized_obs_0,"t h w c -> t c h w")
        normalized_obs_0 = self.resize(normalized_obs_0)
        normalized_obs_0 = rearrange(normalized_obs_0,"t c h w -> t h w c")
        normalized_obs_0 = self.ae_encode(normalized_obs_0.unsqueeze(0)).squeeze(0)
        
        B = context.shape[0]
        context = context.to(ae_device)
        context = rearrange(context,"b t h w c -> (b t) c h w")
        context = self.resize(context)
        context = rearrange(context,"(b t) c h w ->b t h w c",b=B)
        context = self.ae_encode(context)

        sample = super().conditional_sample(
                normalized_obs_0, context, label, (self.num_frames-1, self.image_size, self.image_size, self.image_channel), sampling_timesteps=self.sampling_timesteps, sampling_strategy=self.sampling_strategy
        )
        latent_video = sample[-1]

        # Decode via Autoencoder
        print(f"Latent Video Min: {latent_video.min()}, Latent Video Max: {latent_video.max()}")
        latent_video = latent_video*self.latent_factor+self.latent_midpoint
        latent_video = rearrange(latent_video,"b t h w c -> (b t) c h w")
        print(f"Latent Video Min: {latent_video.min()}, Latent Video Max: {latent_video.max()}")
        
        sampled_video = self.ae.decode(latent_video) 
        print(f"Decoded Video Min: {sampled_video.min()}, Decoded Video Max: {sampled_video.max()}") 
        sampled_video = rearrange(sampled_video,"(b t) c h w -> b t h w c",b=B)
        sampled_video = sampled_video.cpu().detach().numpy()

        return sampled_video
        
    def get_ae_device(self):
        return self.ae.parameters().__next__().device
    
    def load_autoencoder(self, config):
        return Autoencoder.load_from_checkpoint(config["ae_checkpoint"])

    def configure_optimizers(self):
        return Adam(
            list(self.model.parameters()) + list(self.goal_encoder.parameters()), lr=self.lr
        )

    def load_embeddings(self,inst2embed):
        self.instruction2embed = inst2embed
    
    def load_examples(self, example_context):
        self.example_context = example_context
        


def load_error_model(config: dict):
    if config["type"] == "UNet":
        return load_unet_error_model(config["UNet"])
    else:
        raise ValueError("This error model does not exist")

def load_unet_error_model(config: dict):

    error_model = ConditionalUnet3D(
        config["img_channels"],
        config["in_channels"],
        config["time_channels"],
        config["resolutions"],
        config["n_heads"],
        config["use_rotary_emb"],
        config["n_labels"],
        config["n_agents"],
        config["n_frames"],
        config["n_context_frames"],
        config["context_conditioning_type"],
    )

    return error_model