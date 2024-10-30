"""
Agent that interacts with an environment using a diffusion model based planner + action models
Action models available:
- inverse dynamics model
- manual mapping 
"""

import os
import random
import pickle

import blosc
import matplotlib.pyplot as plt
import numpy as np
import torch

from einops import rearrange

from diffusion_nl.imitation_learning.model import ImitationPolicy
from diffusion_nl.ivd_model.model import IVDBabyAI
from diffusion_nl.diffusion_model.model import EDMModel
from diffusion_nl.diffusion_model.utils import transform_sample
from transformers import AutoTokenizer, T5EncoderModel

from minigrid.core.actions import ActionSpace, Actions


class DeterministicAgent:

    def __init__(self, action):
        self.action = action

    def act(self, obs):
        return self.action


class RandomAgent:

    def __init__(self, n_actions):
        self.n_actions = n_actions

    def act(self, obs):
        return [[random.randint(0, self.n_actions - 1) for _ in range(len(obs))]]


class LocalAgent(object):
    """
    Combine a diffusion planner with granularity k with a k-step local policy
    """

    def __init__(self, config):
        self.device = config["device"]
        self.visualize = config["visualize"]
        self.action_space = config["action_space"]
        self.diffusion_planner = self.load_diffusion_planner(config)
        self.policy = self.load_policy(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config["embeddings"]["model"])
        self.encoder_model = T5EncoderModel.from_pretrained(
            config["embeddings"]["model"]
        ).to(self.device)

    def reset(self):
        self.episode_step = 0
        self.labels = None

    def embed_instruction(self, instructions):
        inputs = self.tokenizer(
            instructions, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        model_output = self.encoder_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        embeddings = model_output.last_hidden_state.mean(dim=1)

        return embeddings

    def plan(self, obss):
        if self.episode_step == 0:
            # Embed mission instructions
            print("Embedding instructions")
            self.labels = self.embed_instruction([obs["mission"] for obs in obss]).to(
                self.device
            )

        missions = [obs["mission"] for obs in obss]

        start_obs = torch.stack(
            [torch.tensor(obs["image"], dtype=torch.float) for obs in obss], dim=0
        ).to(self.device)
        # labels = torch.tensor([self.inst2label[obs["mission"]] for obs in obss],dtype=torch.long)
        if self.diffusion_planner.context_type == "agent_id":
            context = (
                torch.tensor([self.action_space] * len(obss)).long().to(self.device)
            )
        elif self.diffusion_planner.context_type == "action_space":
            a_space = ActionSpace(self.action_space)
            legal_actions = [int(a) for a in a_space.get_legal_actions()]
            context = torch.tensor(
                [1 if i in legal_actions else 0 for i in range(len(Actions))]
            ).float().repeat(len(obss), 1).to(self.device)

        elif (
            self.diffusion_planner.context_type == "time"
            or self.diffusion_planner.context_type == "channel"
        ):
            context = (
                torch.tensor(
                    blosc.unpack_array(random.choice(self.context[self.action_space])),
                    dtype=torch.float,
                )
                .unsqueeze(0)
                .repeat(len(obss), 1, 1, 1, 1)
                .to(self.device)
            )
        else:
            raise NotImplementedError(
                f"This context type {self.diffusion_planner.context_type} is not supported"
            )

        # Create a batch for the diffusion model to process
        _, _, xts = self.diffusion_planner.conditional_sample(
            start_obs, context, self.labels
        )
        plans = xts[-1]

        # Visualize plans
        if self.visualize:
            self.visualize_plans(plans, missions)

        return plans

    def act(self, current_states, goals):
        current_states = torch.stack(
            [torch.tensor(obs["image"], dtype=torch.float) for obs in current_states],
            dim=0,
        ).to(self.device)

        current_states = rearrange(current_states, "B H W C -> B C H W")
        goals = rearrange(goals, "B H W C -> B C H W")
        action_probs = self.policy.model(current_states, goals)
        actions = action_probs.argmax(dim=-1)
        return actions

    def visualize_plans(self, plans, missions):
        B, T, _, _, _ = plans.shape
        for i in range(B):
            trajectory = plans[i]
            mission = missions[i]
            images = transform_sample(trajectory)
            for k, image in enumerate(images):
                plt.imshow(image)
                plt.savefig(f"./{mission}_{i}_{k}.png")

    def load_diffusion_planner(self, config):
        filepath = os.path.join(
            config["model_store"],
            config["dm_model_path"],
            config["dm_model_name"],
            config["dm_model_checkpoint"],
        )

        diffusion_planner = StateSpaceDiffusionModel.load_from_checkpoint(
            filepath, map_location=config["device"], config=TEMP_CONFIG
        )
        diffusion_planner.eval()

        return diffusion_planner

    def load_policy(self, config):
        filepath = os.path.join(
            config["model_store"],
            config["policy_path"],
            config["policy_name"],
            config["policy_checkpoint"],
        )

        policy = ImitationPolicy.load_from_checkpoint(
            filepath, map_location=config["device"]
        )
        policy.eval()

        return policy

    def load_examples(self, filepath):
        with open(filepath, "rb") as f:
            self.context = pickle.load(f)


class DiffusionAgent(object):
    def __init__(self, config):
        self.device = config["device"]
        self.visualize = config["visualize"]
        self.action_space = config["action_space"]
        self.n_example_frames = config["n_example_frames"]
        if config["model"]!=None:
            self.diffusion_planner = config["model"].to(self.device)
        else:
            self.diffusion_planner = self.load_diffusion_planner(config)
            
        self.ivd_model = self.load_ivd_model(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config["embeddings"]["model"])
        self.encoder_model = T5EncoderModel.from_pretrained(
            config["embeddings"]["model"]
        ).to(config["device"])

    def reset(self):
        self.plan = []
        self.episode_step = 0
        self.labels = None

    def act(self, obss):
        print("Inside act")
        if self.episode_step == 0:
            # Embed mission instructions
            print("Embedding instructions")
            print(self.device)
            self.labels = self.embed_instruction([obs["mission"] for obs in obss]).to(
                self.device
            )

        missions = [obs["mission"] for obs in obss]
        start_obs = torch.stack(
            [torch.tensor(obs["image"], dtype=torch.float) for obs in obss], dim=0
        ).to(self.device)

        # labels = torch.tensor([self.inst2label[obs["mission"]] for obs in obss],dtype=torch.long)
        if self.diffusion_planner.context_type == "agent_id":
            context = (
                torch.tensor([self.action_space] * len(obss)).long().to(self.device)
            )
        elif self.diffusion_planner.context_type == "action_space":
            a_space = ActionSpace(self.action_space)
            legal_actions = [int(a) for a in a_space.get_legal_actions()]
            context = torch.tensor(
                [1 if i in legal_actions else 0 for i in range(len(Actions))]
            ).float().repeat(len(obss), 1).to(self.device)
        elif (
            self.diffusion_planner.context_type == "time"
            or self.diffusion_planner.context_type == "channel"
        ):
            example_video = blosc.unpack_array(
                random.choice(self.context[self.action_space])
            )
            n_padding_frames = self.n_example_frames - example_video.shape[0]
            if n_padding_frames>0:
                context = np.concatenate(
                    [np.zeros((n_padding_frames, *example_video.shape[1:])), example_video],
                    axis=0,
                )
            else:
                context = example_video
            context = (
                torch.from_numpy(context)
                .float()
                .unsqueeze(0)
                .repeat(len(obss), 1, 1, 1, 1)
                .to(self.device)
            )
        else:
            raise NotImplementedError(
                f"This context type {self.diffusion_planner.context_type} is not supported"
            )

        # Create a batch for the diffusion model to process
        B = len(obss)
        if not self.diffusion_planner.use_instruction:
            self.labels = torch.zeros(B,128).to(self.device)
        if not self.diffusion_planner.use_context:
            context = torch.zeros(B, 1, 8, 8, 3).to(self.device)

        
        print(start_obs.shape, context.shape, self.labels.shape)
        _, xts = self.diffusion_planner.conditional_sample(
            start_obs, context, self.labels
        )
        
        plans = xts

        # Attach starting observations to the plans
        print(plans.shape)
        print(start_obs.shape)
        plans = torch.cat([start_obs.unsqueeze(1), plans], dim=1)

        if self.visualize:
            self.visualize_plans(plans, missions)

        actions = []
        T = plans.shape[1]
        for i in range(T - 1):
            obs_0 = rearrange(plans[:, i], "b h w c -> b c h w")
            obs_1 = rearrange(plans[:, i + 1], "b h w c -> b c h w")

            action = self.ivd_model.predict(obs_0, obs_1)
            actions.append(action)

        self.episode_step += 1

        return actions

    def embed_instruction(self, instructions):
        inputs = self.tokenizer(
            instructions, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        model_output = self.encoder_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        embeddings = model_output.last_hidden_state.mean(dim=1)

        return embeddings

    def visualize_plans(self, plans, missions):
        B, T, _, _, _ = plans.shape
        for i in range(B):
            trajectory = plans[i]
            mission = missions[i]
            images = transform_sample(trajectory)
            for k, image in enumerate(images):
                plt.imshow(image)
                plt.savefig(f"./{mission}_{i}_{k}.png")

    def load_ivd_model(self, config):
        filepath = os.path.join(
            config["model_store"],
            config["ivd_model_path"],
            config["ivd_model_name"],
            config["ivd_model_checkpoint"],
        )
        ivd_model = IVDBabyAI.load_from_checkpoint(filepath, map_location=config["device"])
        ivd_model.eval()

        return ivd_model

    def load_diffusion_planner(self, config):
        filepath = os.path.join(
            config["model_store"],
            config["dm_model_path"],
            config["dm_model_name"],
            config["dm_model_checkpoint"],
        )

        if config["model_type"] == "ddpm":
            diffusion_planner = StateSpaceDiffusionModel.load_from_checkpoint(
                filepath, map_location=config["device"]
            )
        elif config["model_type"] == "edm":
            diffusion_planner = EDMModel.load_from_checkpoint(
                filepath, map_location=config["device"], 
            )
            diffusion_planner.num_steps = config["num_steps"]
        else:
            raise NotImplementedError(
                f"Model type {config['model_type']} is not supported"
            )

        diffusion_planner.eval()

        return diffusion_planner

    def load_examples(self, filepath):
        with open(filepath, "rb") as f:
            self.context = pickle.load(f)
