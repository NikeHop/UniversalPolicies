import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from einops import rearrange

from minigrid.core.actions import ActionSpace

############## Licenses ##############

# This FiLM layer and initialize_parameters fn is taken from https://github.com/mila-iqia/babyai/blob/master/babyai/model.py
# Licensed under the BSD 3-Clause License
#
# BSD 3-Clause License
#
# Copyright (c) 2017, Maxime Chevalier-Boisvert
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


############## Lightning Module ##############


class ImitationPolicy(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters()
        self.lr = config["model"]["lr"]
        self.use_agent_id = config["model"]["use_agent_id"]
        self.action_space = ActionSpace(config["model"]["action_space"])
        self.goal_type = config["goal_type"]
        self.legal_actions = self.action_space.get_legal_actions()

        self.global2local = {action: i for i, action in enumerate(self.legal_actions)}
        self.local2global = {i: action for i, action in enumerate(self.legal_actions)}
        self.model = ImitationPolicyModel(
            config["goal_type"],
            len(self.legal_actions),
            config["model"]["use_agent_id"],
            config["model"]["use_unique_agent_heads"],
        )
        self.loss = nn.CrossEntropyLoss()

    def step(self, batch, step_type):
        obss, goals, global_gt_actions, agent_ids = batch
        local_gt_actions = torch.tensor(
            [self.global2local[a.item()] for a in global_gt_actions]
        ).to(global_gt_actions.device)

        action_probs = self.model(obss, goals, agent_ids)

        # Compute Loss
        loss = self.loss(action_probs, local_gt_actions)
        self.log(f"{step_type}/loss", loss)

        # Determine Accuracy
        acc = (torch.argmax(action_probs, dim=-1) == local_gt_actions).float().mean()
        self.log(f"{step_type}/acc", acc)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "training")
        return loss

    def validation_step(self, batch, batch_idx):
        self.step(batch, "validation")

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)

    def load_embedding_model(self, encoder_model, tokenizer, device):
        self.encoder_model = encoder_model.to(device)
        self.tokenizer = tokenizer

    def act(self, obss, agent_ids, device):
        # Create agent ids
        agent_ids = agent_ids.to(device)

        # Embed Instructions
        if self.goal_type == "language":
            goals = [obs["mission"] for obs in obss]
            inputs = self.tokenizer(
                goals, padding=True, return_tensors="pt", truncation=True
            ).to(device)
            goals = (
                self.encoder_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                .last_hidden_state.mean(dim=1)
                .to(device)
            )
        else:
            raise NotImplementedError(f"Goal type {self.goal_type} not implemented")

        # Current Observation
        obss = torch.stack(
            [torch.from_numpy(obs["image"]).float() for obs in obss], dim=0
        ).to(device)
        obss = rearrange(obss, "B H W C -> B C H W")

        action_probs = self.model(obss, goals, agent_ids)
        local_actions = action_probs.argmax(dim=-1)
        global_actions = torch.tensor(
            [self.local2global[l.item()] for l in local_actions]
        )

        return global_actions


############## Network Modules ##############


class ImitationPolicyModel(nn.Module):
    def __init__(self, goal_type, n_actions, use_agent_id, use_unique_agent_heads):
        super().__init__()
        self.img_encoder = ImageEncoderBabyAI(goal_type)
        self.use_agent_id = use_agent_id
        self.use_unique_agent_heads = use_unique_agent_heads
        self.n_actions = n_actions

        if self.use_agent_id:
            input_dim = 128 * 8 * 8 + 8
        else:
            input_dim = 128 * 8 * 8

        if self.use_unique_agent_heads:
            self.agent_id2actionspace = {}
            self.agentid2policy = nn.ModuleDict()
            for action_space in ActionSpace:
                action_space = ActionSpace(action_space)
                legal_actions = action_space.get_legal_actions()
                local2global = {
                    i: int(action) for i, action in enumerate(legal_actions)
                }
                self.agent_id2actionspace[str(action_space.value)] = local2global
                agent_n_actions = len(local2global)
                self.agentid2policy[str(action_space.value)] = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, agent_n_actions),
                )

        self.policy = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
        )

    def forward(self, obss, goals, agent_ids):
        # Encode Obs with Instructions
        obss = self.img_encoder(obss, goals)

        # Add agent ID to encoded observation
        if self.use_agent_id:
            agent_ids = F.one_hot(agent_ids, num_classes=8)
            obss = torch.cat([obss, agent_ids], dim=1)

        if self.use_unique_agent_heads:
            # Agent-specific policy heads
            predicted_actions = -float("inf") * torch.ones(
                (obss.shape[0], self.n_actions), requires_grad=True
            ).to(obss.device)

            # Apply policy head for each agent id
            for agent_id in self.agent_id2actionspace.keys():
                agent_policy = self.agentid2policy[agent_id]
                local2globalactions = self.agent_id2actionspace[agent_id]
                mask = int(agent_id) == agent_ids
                agent_obss = obss[mask]
                if mask.sum() > 0:
                    action_space_predicted_actions = agent_policy(agent_obss)
                    for local_action, global_action in local2globalactions.items():
                        predicted_actions[
                            mask, global_action
                        ] = action_space_predicted_actions[:, local_action]

        else:
            # One policy head for all agents
            predicted_actions = self.policy(obss)

        return predicted_actions


class FiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=imm_channels,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels,
            out_channels=out_features,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(self.bn2(out))


class ImageEncoderBabyAI(nn.Module):
    def __init__(self, goal="language"):
        super().__init__()

        self.goal = goal

        layers = [
            nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=(3, 3), stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ]

        self.conv = nn.Sequential(*layers)
        self.projection = nn.Linear(8192, 512)
        self.film1 = FiLM(
            in_features=512, out_features=128, in_channels=128, imm_channels=128
        )
        self.film2 = FiLM(
            in_features=512,
            out_features=128,
            in_channels=128,
            imm_channels=128,
        )

        self.pool = nn.MaxPool2d((7, 7), stride=2)

    def forward(self, x, goal):
        # Encode current observation
        x = self.conv(x)
        if self.goal == "obs":
            goal = self.conv(goal).flatten(1, 3)
            goal = self.projection(goal)

        h = self.film1(x, goal)
        x = x + h
        h = self.film2(x, goal)
        x = x + h
        pool = False
        if pool:
            x = self.pool(x)
        return x.flatten(1, 3)


############## Functions ##############


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
