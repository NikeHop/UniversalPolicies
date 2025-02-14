import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.optim import Adam

from minigrid.core.actions import ActionSpace
from universal_policies.utils.networks import InverseDynamicsModelBabyAI


class IVDBabyAI(pl.LightningModule):
    def __init__(self, lr: float, action_space: int):
        super().__init__()

        self.lr = lr
        self.action_space = ActionSpace(action_space)
        self.legal_actions = self.action_space.get_legal_actions()
        self.global2local = {action: i for i, action in enumerate(self.legal_actions)}
        self.local2global = {i: action for i, action in enumerate(self.legal_actions)}
        self.model = InverseDynamicsModelBabyAI(len(self.legal_actions))
        self.loss = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def step(self, batch, step_type):
        obs1, obs2, global_actions = batch
        local_actions = torch.tensor(
            [self.global2local[a.item()] for a in global_actions]
        ).to(global_actions.device)
        action_probs = self.model(obs1, obs2)
        loss = self.loss(action_probs, local_actions)
        self.log(f"{step_type}/loss", loss)
        acc = (torch.argmax(action_probs, dim=1) == local_actions).float().mean()
        self.log(f"{step_type}/acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "training")
        return loss

    def validation_step(self, batch, batch_idx):
        self.step(batch, "validation")

    def predict(self, obs1, obs2):
        action_probs = self.model(obs1, obs2)
        local_actions = torch.argmax(action_probs, dim=1)
        global_actions = torch.tensor(
            [self.local2global[l.item()] for l in local_actions]
        ).to(local_actions.device)
        return global_actions

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)
