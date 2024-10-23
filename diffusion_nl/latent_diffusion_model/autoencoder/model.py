"""
Implementation of AutoencoderKL

Loss: Reconstruction + KL for now 

"""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from einops import rearrange
from torch.distributions import Normal, kl_divergence
from torch.optim import Adam

from diffusion_nl.latent_diffusion_model.autoencoder.data import DATASETID2MEAN, DATASETID2STD
from diffusion_nl.utils.networks import (
    AttentionBlock,
    TimeResidualBlock,
    Swish,
    Downsample,
    Upsample,
)
from diffusion_nl.utils.utils import get_cuda_memory_usage



class Autoencoder(pl.LightningModule):
    def __init__(self, config):

        super().__init__()

        self.config = config
        self.encoder = Encoder(config["encoder"], config["latent_emb_dim"])
        self.decoder = Decoder(config["decoder"], config["latent_emb_dim"])
        self.latent_dim = config["latent_emb_dim"]
        self.loss = nn.MSELoss(reduction="mean")
        self.save_hyperparameters()

    def encode(self, x):
        parameters = self.encoder(x)
        return parameters

    def get_encoding(self, x):
        parameters = self.encode(x)
        mean = parameters[:,:self.latent_dim]
        return mean
    
    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def forward(self, input):
        parameters = self.encode(input)
        mean = parameters[:,:self.latent_dim]
        log_std = parameters[:,self.latent_dim:]
        log_std = torch.clamp(log_std,-30,20)
        std = torch.exp(log_std)
        print(mean.shape,std.shape)
        z = mean + torch.randn_like(std)*std
        dec = self.decode(z)
        return dec, parameters

    def training_step(self, batch, batch_idx):
        imgs = batch
        imgs = rearrange(imgs, "b h w c -> b c h w")
        dec, parameters = self(imgs)
        kl_loss = self.kl_loss(parameters)
        loss = self.loss(dec, imgs) + self.config["kl_weight"]*kl_loss
        self.log("training/kl_loss", kl_loss)
        self.log("training/loss", loss)
        return loss

    def kl_loss(self, parameters):
        mean = parameters[:,:self.latent_dim]
        log_std = parameters[:,self.latent_dim:]
        log_std = torch.clamp(log_std,-30,20)
        target_normal = Normal(mean, torch.exp(log_std))
        standard_normal = Normal(torch.zeros_like(mean), torch.ones_like(log_std))
        
        kl_loss = kl_divergence(target_normal,standard_normal)

        return kl_loss.mean()
    
    def validation_step(self, batch, batch_idx):
        imgs = batch
        imgs = rearrange(imgs, "b h w c -> b c h w")
        dec, parameters = self(imgs)
        kl_loss = self.kl_loss(parameters)
        loss = self.loss(dec, imgs) + self.config["kl_weight"]*kl_loss
        self.log("validation/kl_loss", kl_loss)
        self.log("validation/loss", loss)
         
        if batch_idx==0:
            self.visualize(imgs[0], dec[0])

        return loss

    def visualize(self, sample, reconstruction):
        mean = 127.5
        std = 127.5
        sample = self.transform(sample,mean,std)
        reconstruction = self.transform(reconstruction,mean,std)
        self.logger.experiment.log(
            {
                f"Sample Reconstruction": [
                    wandb.Image(
                        sample,
                        caption=f"sample",
                    ),
                    wandb.Image(
                        reconstruction,
                        caption=f"reconstruction",
                    ),
                ]
            }
        )

    def configure_optimizers(self):
        optimizer = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.config["lr"],
        )
        return optimizer

    def transform(self, img, mean, std):
        img = (img*std + mean).permute(1,2,0).int().cpu().numpy()
        return img


class Encoder(nn.Module):

    def __init__(self, config, latent_emb_dim):

        super().__init__()

        # Init Conv
        self.init_conv = nn.Conv2d(3, config["n_channels"], kernel_size=3, padding=1)

        down = []
        out_channels = in_channels = config["n_channels"]

        for k, multiplier in enumerate(config["resolutions"]):
            out_channels = in_channels * multiplier

            resnet_block_1 = TimeResidualBlock(
                in_channels, out_channels, 1, n_groups=config["n_groups"], d2=True
            )
            # Normalization + Activation + Conv(3,3) + Normalization + Activation + Dropout + Conv(3,3)

            # resnet_block_2 = TimeResidualBlock(out_channels, out_channels, 1, n_groups=config["n_groups"], d2=True)
            # spatial_attention = AttentionBlock(
            #     out_channels, n_heads=config["n_heads"], n_groups=config["n_groups"]
            # )

            block = nn.ModuleList([resnet_block_1])

            downsampling = (
                Downsample(out_channels, d2=True)
                if k < len(config["resolutions"]) - 1
                else nn.Identity()
            )
            down.append(nn.ModuleList([block, downsampling]))
            in_channels = out_channels

        self.down = nn.ModuleList(down)

        # Middle Block
        # self.mid_resnet_prev = TimeResidualBlock(out_channels, out_channels, 1, n_groups=config["n_groups"], d2=True)

        # self.mid_spatial_attention = AttentionBlock(
        #     out_channels, n_heads=config["n_heads"], n_groups=config["n_groups"]
        # )

        # self.mid_resnet_after = TimeResidualBlock(
        #     out_channels, out_channels, 1, n_groups=config["n_groups"], d2=True
        # )

        # End
        self.final_conv = nn.ModuleList(
            [
                nn.GroupNorm(config["n_groups"], out_channels),
                Swish(),
                nn.Conv2d(out_channels, 2*latent_emb_dim, kernel_size=3, padding=1),
            ]
        )

    def forward(self, x):
        x = self.init_conv(x)

        for block, downsample in self.down:
            for resnet_block in block:
                x = resnet_block(x, None)
            x = downsample(x)

        # x = self.mid_resnet_prev(x, None)
        # x = self.mid_spatial_attention(x, None)
        # x = self.mid_resnet_after(x, None)

        for block in self.final_conv:
            x = block(x)

        return x


class Decoder(nn.Module):

    def __init__(self, config, latent_emb_dim):

        super().__init__()
        # Init-Conv
        self.conv_init = nn.Conv2d(
            latent_emb_dim, config["n_channels"], kernel_size=3, padding=1
        )

        # Middle Section
        # self.mid_resnet_prev = TimeResidualBlock(
        #     config["n_channels"], config["n_channels"], 1, n_groups=config["n_groups"], d2=True
        # )

        # self.mid_spatial_attention = AttentionBlock(
        #     config["n_channels"], n_heads=config["n_heads"], n_groups=config["n_groups"]
        # )

        # self.mid_resnet_after = TimeResidualBlock(
        #     config["n_channels"], config["n_channels"], 1, n_groups=config["n_groups"], d2=True
        # )

        # Upsampling
        in_channels = out_channels = config["n_channels"]
        self.up = nn.ModuleList([])
        for k, multiplier in enumerate(reversed(config["resolutions"])):
            print(out_channels)
            out_channels = in_channels // multiplier
            resnet_block_1 = TimeResidualBlock(
                in_channels, out_channels, 1, n_groups=config["n_groups"], d2=True
            )
            # resnet_block_2 = TimeResidualBlock(
            #     out_channels, out_channels, 1, n_groups=config["n_groups"], d2=True
            # )

            # attention = AttentionBlock(
            #     out_channels, n_groups=config["n_groups"], n_heads=config["n_heads"]
            # )

            block = nn.ModuleList([resnet_block_1])

            upsample = (
                Upsample(out_channels, d2=True)
                if k < len(config["resolutions"]) - 1
                else nn.Identity()
            )
            self.up.append(nn.ModuleList([block, upsample]))
            in_channels = out_channels

        # End
        self.final_conv = nn.ModuleList(
            [
                nn.GroupNorm(config["n_groups"], out_channels),
                Swish(),
                nn.Conv2d(out_channels, 3, kernel_size=3, padding=1),
            ]
        )

        if config["tanh"]:
            self.final_conv.append(nn.Tanh())

    def forward(self, x):
        # Init Conv
        x = self.conv_init(x)

        # Middle Block
        # x = self.mid_resnet_prev(x, None)
        # x = self.mid_spatial_attention(x, None)
        # x = self.mid_resnet_after(x, None)

        # Upsampling
        for block, upsample in self.up:
            for layer in block:
                x = layer(x, None)
            x = upsample(x)

        # End
        for block in self.final_conv:
            x = block(x)

        return x
