"""
Basic Conditional Diffusion Model for the BabyAI state space
"""

import os 
import random 

import blosc 
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from diffusion_nl.diffusion_model.utils import transform_sample, state2img
from diffusion_nl.utils.models import EDM 
from diffusion_nl.utils.networks import ConditionalUnet3DDhariwal

from minigrid.core.actions import ActionSpace, Actions

class EDMModel(EDM):

    def __init__(self, env, config: dict={}):
        """
        Args:
            env: Environment for visualization
            config: Configuration dictionary
        """
        # Logging config
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
        """
        Sample an example trajectory conditioned on the context
        """
        samples, missions, context_types = self.create_conditional_samples(self.n_samples)

        for sample, mission, context_type in zip(samples,missions,context_types):
            obs_0, sample = sample 
            starting_state = state2img(obs_0[0].cpu().numpy())
            traj_states = transform_sample(sample[0])
            traj_states = [starting_state] + traj_states

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
        """
        Create n_samples of conditional samples

        Args:
            n_samples: Number of samples to create

        Returns:
            samples: List of tuples - (obs_0: torch.Tensor, generated_trajectory: torch.Tensor)
            missions: List of str - instructions
            action_spaces: List of ints - action space that was conditioned on
        """
        samples = []
        action_spaces = []
        missions = []
        attempts = 0 
        while len(missions)<n_samples:
            # Sample a starting state
            obs = self.env.reset()[0]
            mission = obs["mission"]

            # Check whether embedding is available
            if mission not in self.instruction2embed:
                attempts += 1
                if attempts > 100:
                    break
                continue
            
            # Prepare model input
            obs_0 = torch.tensor(obs["image"],dtype=torch.float).unsqueeze(0)
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
            action_spaces.append(k)

        return samples, missions, action_spaces

    def conditional_sample(self, obs_0, context, label):
        """
        Sample a trajectory conditioned on the context

        Args:
            obs_0 (torch.Tensor): Current environment observation

        Returns:
            tuple: (obs_0: torch.Tensor, generated_trajectory: torch.Tensor)
        """

        normalized_obs_0 = (obs_0 - self.mean) / self.std
        if self.context_type=="time" or self.context_type=="channel":
            context = (context - self.mean) / self.std
            
        sample = super().conditional_sample(
                normalized_obs_0, context, label, (self.num_frames-1, self.image_size, self.image_size, self.image_channel)
            )
        sample = sample*self.std + self.mean
    
        return (obs_0,sample)

    def load_embeddings(self,inst2embed):
        self.instruction2embed = inst2embed
    
    def load_examples(self, example_context):
        self.example_context = example_context


def load_model(config: dict, model_type: str, use_context: bool, use_instruction: bool):

    if model_type=="edm":
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
