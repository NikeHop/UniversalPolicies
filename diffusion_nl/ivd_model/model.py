import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.optim import Adam

from minigrid.core.actions import ActionSpace
from diffusion_nl.utils.networks import InverseDynamicsModelBabyAI, InverseDynamicsModelCalvin


class IVDBabyAI(pl.LightningModule):
    def __init__(self, lr: float, action_space: int):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.action_space = ActionSpace(action_space)
        self.legal_actions = self.action_space.get_legal_actions()
        self.global2local = {action: i for i, action in enumerate(self.legal_actions)}
        self.local2global = {i: action for i, action in enumerate(self.legal_actions)}
        print(f"################# {self.action_space} #################")
        print(self.global2local)
        print(self.local2global)
        self.model = InverseDynamicsModelBabyAI(len(self.legal_actions))
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        obs1, obs2, global_actions = batch
        local_actions = torch.tensor(
            [self.global2local[a.item()] for a in global_actions]
        ).to(global_actions.device)
        action_probs = self.model(obs1, obs2)
        loss = self.loss(action_probs, local_actions)
        self.log("training/loss", loss)

        # Determine Accuracy
        acc = (torch.argmax(action_probs, dim=1) == local_actions).float().mean()
        self.log("training/acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        obs1, obs2, global_actions = batch
        local_actions = torch.tensor(
            [self.global2local[a.item()] for a in global_actions]
        ).to(global_actions.device)
        action_probs = self.model(obs1, obs2)

        # Determine Accuracy
        acc = (torch.argmax(action_probs, dim=1) == local_actions).float().mean()
        self.log("validation/acc", acc)

    def predict(self, obs1, obs2):
        action_probs = self.model(obs1, obs2)
        local_actions = torch.argmax(action_probs, dim=1)
        global_actions = torch.tensor(
            [self.local2global[l.item()] for l in local_actions]
        ).to(local_actions.device)
        return global_actions

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)


class IVDCalvin(pl.LightningModule):

    def __init__(self, lr: float, action_dim: int):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.model = InverseDynamicsModelCalvin(action_dim)
        self.loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        obs1, obs2, gt_actions = batch
        pred_actions = self.model(obs1, obs2)
        loss = self.loss(pred_actions, gt_actions)
        self.log("training/loss", loss)

        # Determine Accuracy
        action_acc = torch.abs(pred_actions[:,:-1] - gt_actions[:,:-1]).float().mean()
        gripper_acc = ((pred_actions[:,-1]>0) == (gt_actions[:,-1]>0)).float().mean()
        
        self.log("training/gripper_acc", gripper_acc)
        self.log("training/action_acc", action_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        obs1, obs2, gt_actions = batch
        pred_actions = self.model(obs1, obs2)
        loss = self.loss(pred_actions, gt_actions)
        self.log("validation/loss", loss)

        # Determine Accuracy
        action_acc = torch.abs(pred_actions[:,:-1] - gt_actions[:,:-1]).float().mean()
        gripper_acc = ((pred_actions[:,-1]>0) == (gt_actions[:,-1]>0)).float().mean()
        
        self.log("validation/gripper_acc", gripper_acc)
        self.log("validation/action_acc", action_acc)

    def predict(self, obs1, obs2):
        pred_actions = self.model(obs1, obs2)
        return pred_actions

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)
