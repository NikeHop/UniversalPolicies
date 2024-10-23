"""
Basic Conditional Diffusion Model for the BabyAI state space
"""
import os 
import random 

import blosc 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
import numpy as np

from einops import rearrange

from diffusion_nl.diffusion_model.utils import transform_sample, state2img
from diffusion_nl.utils.models import BaseDiffusion, EDM 
from diffusion_nl.utils.networks import ConditionalUnet3D, ConditionalUnet3DDhariwal

from minigrid.core.actions import ActionSpace, Actions

class StateSpaceDiffusionModel(BaseDiffusion):

    def __init__(self, env, config: dict={}):
        """
        T: number of diffusion steps
        beta_start: starting value of beta
        beta_end: value of beta at last diffusion step
        """
        self.debug = config["debug"]
        self.image_directory = config["image_directory"]
        

        # Eval config
        self.n_samples = config["eval"]["n_samples"]
        self.num_frames = config["eval"]["n_frames"]
        self.context_frames = config["eval"]["n_context_frames"]
        self.context_type = config["error_model"]["UNet"]["context_conditioning_type"]

        # Data Config
        self.image_size = config["image_size"]
        self.image_channel = config["image_channel"]

        model = load_model(config["error_model"])
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
        self.save_hyperparameters()

        # Environment for visualization
        self.env = env 
        

    def training_step(self, batch, batch_idx):
        x0, mask, context, labels = batch
        B = x0.shape[0]
        loss = super().training_step(x0, mask, context, labels, B)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, mask, context, labels = batch
        B = x0.shape[0]
        super().validation_step(x0, mask, context, labels, B)

    def on_validation_epoch_end(self):
        if not self.debug:
            samples, missions, context_types = self.create_conditional_samples(self.n_samples)

            for sample, mission, context_type in zip(samples,missions,context_types):
                obs_0, _, sample = sample 

                starting_state = state2img(obs_0[0].cpu().numpy())
                traj_states = transform_sample(sample[-1][0])

                traj_states = [starting_state] + traj_states

                # Determine min and max of the trajectory
                total_min = -float('inf')
                total_max = float('inf')
                for s in sample[0][0]:
                    min = torch.min(s)
                    max = torch.max(s)
                    if min<total_min:
                        total_min = min
                    if max>total_max:
                        total_max = max

                self.log("state_min",min)
                self.log("state_max",max)

                # Log to wandb
                self.logger.experiment.log(
                    {
                        f"Sample Trajectory {context_type}": [
                            wandb.Image(
                                    state,
                                caption=f"{mission}",
                            )
                        for state in traj_states]
                    }
                )

                # Log locally
                for i, state in enumerate(traj_states):
                    plt.imshow(state)
                    plt.axis("off")
                    filepath = os.path.join(self.image_directory,f"{mission}_{i}.png")
                    plt.savefig(filepath)

    def predict_x0(self, xt, t, obs_0, context, label):
        prediction = self.model(xt, t, obs_0, context, label)
        prediction = rearrange(prediction, "b c t h w -> b t h w c")
        return prediction  

    def create_conditional_samples(self, n_samples):
        samples = []
        context_types = []
        missions = []

        for i in range(n_samples):
            
            obs = self.env.reset()[0]
            attempts = 0
            while obs["mission"] not in self.instruction2embed:
                obs = self.env.reset()[0]
                attempts += 1
                if attempts>1000:
                    return 

            obs_0 = torch.tensor(obs["image"],dtype=torch.float).unsqueeze(0)
            print(self.env, obs_0.shape)

            mission = obs["mission"]
            missions.append(mission)

            label = self.instruction2embed[mission]
            k = random.choice(list(self.example_context.keys()))
            
            if self.context_type == "agent_id":
                context = torch.tensor(k, dtype=torch.long).unsqueeze(0)
            elif self.context_type == "action_space":
                action_space = ActionSpace(k)
                legal_actions = [int(a) for a in action_space.get_legal_actions()]
                legal_actions = [1 if i in legal_actions else 0 for i in range(len(Actions))]
                context = torch.tensor(legal_actions).float().reshape(1,-1)
            elif self.context_type == "time" or self.context_type == "channel":
                context = blosc.unpack_array(random.choice(self.example_context[k]))
                n_padding_frames = self.context_frames - context.shape[0]
                if n_padding_frames>0:
                    padding = np.zeros((n_padding_frames,*context.shape[1:]))
                    context = np.concatenate([padding,context],axis=0)
                context = torch.tensor(context, dtype=torch.float).unsqueeze(0)
            else:
                raise NotImplementedError(f"Context type {self.context_type} not implemented")

            sample = self.conditional_sample(
                obs_0, context, label,
            )
            samples.append(sample)
            context_types.append(k)

        return samples, missions, context_types

    def conditional_sample(self, obs_0, context, label):
        normalized_obs_0 = (obs_0 - self.mean) / self.std
        if self.context_type=="time" or self.context_type=="channel":
            context = (context - self.mean) / self.std
        sample = super().conditional_sample(
                normalized_obs_0, context, label, (self.num_frames-1, self.image_size, self.image_size, self.image_channel), sampling_timesteps=self.sampling_timesteps, sampling_strategy=self.sampling_strategy
            )
        return (obs_0,label,sample)

    def load_embeddings(self,inst2embed):
        self.instruction2embed = inst2embed
    
    def load_examples(self, example_context):
        self.example_context = example_context
        


class EDMModel(EDM):

    def __init__(self, env, config: dict={}):
        """
        T: number of diffusion steps
        beta_start: starting value of beta
        beta_end: value of beta at last diffusion step
        """
        self.debug = config["debug"]
        self.image_directory = config["image_directory"]
        
        # Eval config
        self.n_samples = config["eval"]["n_samples"]
        self.num_frames = config["eval"]["n_frames"]
        self.context_frames = config["eval"]["n_context_frames"]
        self.context_type = config["error_model"]["context_conditioning_type"]

        # Data Config
        self.image_size = config["image_size"]
        self.image_channel = config["image_channel"]

        # Model config 
        self.use_instruction = config["use_instruction"]
        self.use_context = config["use_context"]

        model = load_model(config["error_model"], config["model_type"], config["use_context"], config["use_instruction"])

        super().__init__(
            model,
            config["lr"],
            config["P_mean"],
            config["P_std"],
            config["sigma_data"],
            config["num_steps"],
            config["min_sigma"],
            config["max_sigma"],
            config["rho"],
            config["mean"],
            config["std"]
            
        )
        self.save_hyperparameters()

        # Environment for visualization
        self.env = env 
        

    def training_step(self, batch, batch_idx):
        x0, mask, context, labels = batch

        if not self.use_instruction:
            labels = torch.zeros_like(labels)
        
        if not self.use_context:
            context = torch.zeros_like(context)

        loss = super().training_step(x0, context, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, mask, context, labels = batch

        if not self.use_instruction:
            labels = torch.zeros_like(labels)
        
        if not self.use_context:
            context = torch.zeros_like(context)
            
        super().validation_step(x0, context, labels)

    def on_validation_epoch_end(self):
        if not self.debug:
            samples, missions, context_types = self.create_conditional_samples(self.n_samples)

            for sample, mission, context_type in zip(samples,missions,context_types):
                obs_0, _, sample = sample 

                starting_state = state2img(obs_0[0].cpu().numpy())
                traj_states = transform_sample(sample[0])

                traj_states = [starting_state] + traj_states

                # Determine min and max of the trajectory
                total_min = -float('inf')
                total_max = float('inf')
                for s in sample[0][0]:
                    min = torch.min(s)
                    max = torch.max(s)
                    if min<total_min:
                        total_min = min
                    if max>total_max:
                        total_max = max

                self.log("state_min",min)
                self.log("state_max",max)

                # Log to wandb
                self.logger.experiment.log(
                    {
                        f"Sample Trajectory {context_type}": [
                            wandb.Image(
                                    state,
                                caption=f"{mission}",
                            )
                        for state in traj_states]
                    }
                )

                # Log locally
                for i, state in enumerate(traj_states):
                    plt.imshow(state)
                    plt.axis("off")
                    filepath = os.path.join(self.image_directory,f"{mission}_{i}.png")
                    plt.savefig(filepath)

    def create_conditional_samples(self, n_samples):
        samples = []
        context_types = []
        missions = []

        for i in range(n_samples):
            
            obs = self.env.reset()[0]
            attempts = 0
            while obs["mission"] not in self.instruction2embed:
                obs = self.env.reset()[0]
                attempts += 1
                if attempts>1000:
                    return 

            obs_0 = torch.tensor(obs["image"],dtype=torch.float).unsqueeze(0)
            print(self.env, obs_0.shape)

            mission = obs["mission"]
            missions.append(mission)

            label = self.instruction2embed[mission]
            k = random.choice(list(self.example_context.keys()))
            
            if self.context_type == "agent_id":
                context = torch.tensor(k, dtype=torch.long).unsqueeze(0)
            elif self.context_type == "action_space":
                action_space = ActionSpace(k)
                legal_actions = [int(a) for a in action_space.get_legal_actions()]
                legal_actions = [1 if i in legal_actions else 0 for i in range(len(Actions))]
                context = torch.tensor(legal_actions).float().reshape(1,-1)
            elif self.context_type == "time" or self.context_type == "channel":
                context = blosc.unpack_array(random.choice(self.example_context[k]))
                n_padding_frames = self.context_frames - context.shape[0]
                padding = np.zeros((n_padding_frames,*context.shape[1:]))
                context = np.concatenate([padding,context],axis=0)
                context = torch.tensor(context, dtype=torch.float).unsqueeze(0)
            else:
                raise NotImplementedError(f"Context type {self.context_type} not implemented")

            sample = self.conditional_sample(
                obs_0, context, label,
            )
            
            samples.append(sample)
            context_types.append(k)

        
        return samples, missions, context_types

    def conditional_sample(self, obs_0, context, label):
        normalized_obs_0 = (obs_0 - self.mean) / self.std
        if self.context_type=="time" or self.context_type=="channel":
            context = (context - self.mean) / self.std
            
        sample = super().conditional_sample(
                normalized_obs_0, context, label, (self.num_frames-1, self.image_size, self.image_size, self.image_channel)
            )
        sample = sample*self.std + self.mean
    
        return (obs_0,label,sample)

    def load_embeddings(self,inst2embed):
        self.instruction2embed = inst2embed
    
    def load_examples(self, example_context):
        self.example_context = example_context


def load_model(config: dict, model_type: str, use_context: bool, use_instruction: bool):

    if model_type=="ddpm":

        model = ConditionalUnet3D(
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

    elif model_type=="edm":

        model = ConditionalUnet3DDhariwal(
            config["img_channels"],
            config["in_channels"],
            config["time_channels"],
            config["resolutions"],
            config["n_heads"],
            config["use_rotary_emb"],
            config["label_dim"],
            config["label_dropout"],
            use_context,
            use_instruction,
            config["n_agents"],
            config["n_frames"],
            config["n_context_frames"],
            config["context_conditioning_type"],
        )

    else:
        raise NotImplementedError("Model type not implemented")

    return model
