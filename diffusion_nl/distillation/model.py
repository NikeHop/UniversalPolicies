from copy import deepcopy
import random

import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from torch.optim import Adam

from einops import rearrange

from diffusion_nl.diffusion_model.utils import transform_sample, state2img
from diffusion_nl.diffusion_model.model import load_error_model, StateSpaceDiffusionModel
from diffusion_nl.utils.models import BaseDiffusion


class DistillationModel(pl.LightningModule):

    def __init__(self, teacher,  config):

        super().__init__()

        # Hyperparameters
        self.lr = config["lr"]

        # Teacher
        self.teacher = teacher
        self.teacher.eval()

        # Loss
        self.loss = nn.MSELoss(reduce="mean")

        # Take information from the teacher model
        self.alphas_bars = teacher.alpha_bars
        self.teacher_step_size = (
            teacher.sampling_timesteps[1] - teacher.sampling_timesteps[0]
        )
        self.sampling_steps = [
            step for i, step in enumerate(self.teacher.sampling_timesteps) if i % 2 == 0
        ]

        self.shape = (
            self.teacher.num_frames - 1,
            self.teacher.image_size,
            self.teacher.image_size,
            self.teacher.image_channel,
        )

        # Copy the teacher
        self.student_model = deepcopy(self.teacher)

        if self.student_model.model_type == "error":
            self.student_model.model = load_error_model(config["error_model"])
            self.student_model.model_type = "x"

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, mask, context, labels = batch
        loss = self.get_loss(x, mask, context, labels)
        self.log("training/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, context, labels = batch
        with torch.no_grad():
            loss = self.get_loss(x, mask, context, labels)
        self.log("validation/loss", loss)

    def on_validation_epoch_end(self):
        # Prepare Input
        samples, context_types = self.student_model.create_conditional_samples(1)

        for sample, context_type in zip(samples,context_types):
            obs_0, mission, sample = sample

            starting_state = state2img(obs_0[0].cpu().numpy())
            traj_states = transform_sample(sample[-1][0])

            traj_states = [starting_state] + traj_states

            # Determine min and max of the trajectory
            total_min = -float("inf")
            total_max = float("inf")
            for s in sample[0][0]:
                min = torch.min(s)
                max = torch.max(s)
                if min < total_min:
                    total_min = min
                if max > total_max:
                    total_max = max

            self.log("state_min", min)
            self.log("state_max", max)

            # Log to wandb
            self.logger.experiment.log(
                {
                    f"Sample Trajectory {context_type}": [
                        wandb.Image(
                            state,
                            caption=f"{mission}",
                        )
                        for state in traj_states
                    ]
                }
            )

    def get_loss(self, x, mask, context, labels):
        # Preprocess the data

        obs_0, x = x[:, 0], x[:, 1:]
        B = x.shape[0]

        # Sample an error & timestep
        err = torch.randn_like(x)
        t = torch.tensor(random.sample(self.student_model.sampling_timesteps, k=B)).type_as(x).long()
        t_1 = t - self.teacher_step_size
        t_2 = t_1 - self.teacher_step_size

        # Perform forward diffusion
        xt = self.teacher.forward_diffusion(x, err, t)

        # Perform two steps of DDIM
        xtt = self.teacher.backward_step(
            xt,
            t,
            t_1,
            obs_0,
            context,
            labels,
            self.shape,
            sampling_strategy="ddim"
        )
        xttt = self.teacher.backward_step(
            xtt,
            t_1,
            t_2,
            obs_0,
            context,
            labels,
            self.shape,
            sampling_strategy="ddim"
        )

        # Compute target
        target = self.get_target(xt, xttt, t, t_2)

        # Predict x from the student model
        prediction = self.student_model.predict_x0(xt, t, obs_0, context, labels)

        # Determine loss
        loss = self.loss(target, prediction)
        return loss 

    def get_target(self, xt, xttt, t, t_2):
        alpha_bar_t = self.alphas_bars[t].reshape(-1,1,1,1,1)
        alpha_bar_t_2 = self.alphas_bars[t_2].reshape(-1,1,1,1,1)

        scale = torch.sqrt(1 - alpha_bar_t_2) / torch.sqrt(
            1 - alpha_bar_t
        )

        resize = torch.sqrt(alpha_bar_t) * (1 - scale)
        target = resize * (xttt - scale * xt)
        return target

    def configure_optimizers(self):
        return Adam(self.student_model.parameters(), lr=self.lr)
