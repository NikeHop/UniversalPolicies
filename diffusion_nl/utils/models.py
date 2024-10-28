from einops import rearrange

import pytorch_lightning as pl
import torch

from torch.optim import Adam

########### License ###########

# The EDM class is taken from https://github.com/NVlabs/edm
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES
#
# You may share and adapt this code for non-commercial purposes as long as you provide attribution,
# indicate any changes made, and share derivative works under the same license.
#
# License: https://creativecommons.org/licenses/by-nc-sa/4.0/

########### License ###########


class EDM(pl.LightningModule):

    def __init__(
        self,
        model,
        lr,
        P_mean,
        P_std,
        sigma_data,
        num_steps,
        min_sigma,
        max_sigma,
        rho,
        mean,
        std,
    ):

        super().__init__()

        self.model = model
        self.lr = lr
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.num_steps = num_steps
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.rho = rho
        self.mean = mean
        self.std = std

        self.save_hyperparameters()

    def training_step(self, video, context, labels):
        loss = self.get_loss(video, context, labels)
        self.log("training/loss", loss)
        return loss
    
    def validation_step(self, video, context, labels):
        loss = self.get_loss(video, context, labels)
        self.log("validation/loss", loss)
        return loss

    def get_loss(self, video, context, labels):
        obs_0, video = video[:, 0], video[:, 1:]

        rnd_normal = torch.randn([video.shape[0], 1, 1, 1, 1], device=video.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        n = torch.randn_like(video) * sigma

        D_video_n = self.get_prediction(video + n, sigma, obs_0, context, labels)
        loss = weight * ((D_video_n - video) ** 2)

        return loss.mean()

    def get_prediction(self, x, sigma, obs_0, context, labels):
        sigma = sigma.reshape(-1, 1, 1, 1, 1)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(c_in * x, c_noise.flatten(), obs_0, context, labels)
        F_x = rearrange(F_x, "b c t h w -> b t h w c")

        D_x = c_skip * x + c_out * F_x

        return D_x


    def conditional_sample(self, obs_0, context, labels, shape):

        # Send to device
        obs_0 = obs_0.to(self.device)
        context = context.to(self.device)
        labels = labels.to(self.device)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, device=self.device)
        t_steps = (
            self.max_sigma ** (1 / self.rho)
            + step_indices
            / (self.num_steps - 1)
            * (self.min_sigma ** (1 / self.rho) - self.max_sigma ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        B = obs_0.shape[0]
        latents = torch.randn(B, *shape).to(self.device)
        x_next = latents * t_steps[0]

        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            # Disabled stochastic sampling
            # gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            # t_hat = self.model.round_sigma(t_cur + gamma * t_cur)

            x_hat = x_cur  # + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
            t_hat = t_cur

            # Euler step.
            denoised = self.get_prediction(
                x_hat, t_hat.reshape(-1), obs_0, context, labels
            )

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                denoised = self.get_prediction(
                    x_next, t_next.reshape(-1), obs_0, context, labels
                )
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)
