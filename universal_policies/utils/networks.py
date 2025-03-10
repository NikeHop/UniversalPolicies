import math

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

########### Licenses ###########

# The ConditionalUnet3DDhariwal class is taken from https://github.com/NVlabs/edm
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES
#
# You may share and adapt this code for non-commercial purposes as long as you provide attribution,
# indicate any changes made, and share derivative works under the same license.
#
# License: https://creativecommons.org/licenses/by-nc-sa/4.0/

# The components of the UNet are adapted from https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# Licensed under the MIT License
#
# MIT License
#
# Copyright (c) 2020 Varuna Jayasiri
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

########### Licenses ###########


class InverseDynamicsModelBabyAI(nn.Module):
    """
    Inverse Dynamics Model for BabyAI environment.

    This model takes two input images and predicts the action that leads from the first image to the second image.

    Args:
        n_actions (int): Number of possible actions.

    Attributes:
        initial_conv (nn.Conv2d): Initial convolutional layer.
        res_block1 (ResidualBlock): Residual block.
        fc (nn.Linear): Fully connected layer.
    """

    def __init__(self, n_actions):
        super(InverseDynamicsModelBabyAI, self).__init__()
        self.initial_conv = nn.Conv2d(
            2 * 3, 128, kernel_size=3, stride=1, padding=1
        )  # Assuming input has 3 channels
        self.res_block1 = ResidualBlock(128)
        self.fc = nn.Linear(128, n_actions)  # MLP layer

    def forward(self, x0, x1):
        """
        Forward pass of the Inverse Dynamics Model.

        Args:
            x0 (torch.Tensor): First input image.
            x1 (torch.Tensor): Second input image.

        Returns:
            torch.Tensor: Predicted action.

        """
        x = torch.cat([x0, x1], dim=1)
        x = F.relu(self.initial_conv(x))
        x = self.res_block1(x)
        x = torch.mean(x, dim=(2, 3))  # Mean-pooling across all pixel locations
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    """
    A residual block module that performs residual learning in a neural network.

    Args:
        in_channels (int): The number of input channels.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.

    """

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual
        return F.relu(x)


# Adapted from https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# MIT License
# Copyright (c) 2020 Varuna Jayasiri


class AttentionBlock(nn.Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(
        self,
        n_channels: int,
        n_heads: int = 1,
        d_k: int = None,
        n_groups: int = 32,
        use_rotary_emb: bool = False,
    ):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Rotary Embeddings
        self.use_rotary_emb = use_rotary_emb
        self.rotary_emb = RotaryEmbedding(dim=d_k)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k**-0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t=None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Rotate query and key
        if self.use_rotary_emb:
            q = rearrange(q, "b s h d -> b h s d")
            q = self.rotary_emb.rotate_queries_or_keys(q)
            q = rearrange(q, "b h s d -> b s h d")

            k = rearrange(k, "b s h d -> b h s d")
            k = self.rotary_emb.rotate_queries_or_keys(k)
            k = rearrange(k, "b h s d -> b s h d")

        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.reshape(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res


class TimeResidualBlock(nn.Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        n_groups: int = 32,
        dropout: float = 0.1,
        d2: bool = True,
    ):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        * `d2` whether the convolution shoul be 2d or 3d
        """
        super().__init__()
        self.d2 = d2

        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()

        if self.d2:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
            )
        else:
            self.conv1 = nn.Conv3d(
                in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
            )

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()

        if self.d2:
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
            )
        else:
            self.conv2 = nn.Conv3d(
                out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
            )

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            if self.d2:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
            else:
                self.shortcut = nn.Conv3d(
                    in_channels, out_channels, kernel_size=(1, 1, 1)
                )
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        In d2 case:
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`

        In d3 case:
        * `x` has shape `[batch_size, in_channels, frames, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))

        # Add time embeddings
        if self.d2:
            if t != None:
                h += self.time_emb(self.time_act(t))[:, :, None, None]
        else:
            h += self.time_emb(self.time_act(t))[:, :, None, None, None]

        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """
    ### Down block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        d2: bool,
    ):
        super().__init__()
        self.res = TimeResidualBlock(in_channels, out_channels, time_channels, d2=d2)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    ### Up block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        d2: bool,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = TimeResidualBlock(
            in_channels + out_channels, out_channels, time_channels, d2
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int, d2: bool):
        super().__init__()
        self.res1 = TimeResidualBlock(n_channels, n_channels, time_channels, d2)
        self.attn = AttentionBlock(n_channels)
        self.res2 = TimeResidualBlock(n_channels, n_channels, time_channels, d2)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels, d2: bool = True):
        super().__init__()
        if d2:
            self.conv = nn.ConvTranspose2d(
                n_channels, n_channels, (4, 4), (2, 2), (1, 1)
            )
        else:
            self.conv = nn.ConvTranspose3d(
                n_channels, n_channels, (1, 4, 4), (1, 2, 2), (0, 1, 1)
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor = None):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels, d2: bool = True):
        super().__init__()

        if d2:
            self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))
        else:
            self.conv = nn.Conv3d(
                n_channels, n_channels, (1, 3, 3), (1, 2, 2), (0, 1, 1)
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor = None):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Swish(nn.Module):
    """
    ### Swish actiavation function

    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb


class SpatialAttention(AttentionBlock):
    def __init__(
        self, n_channels: int, n_heads: int, d_k: int = None, n_groups: int = 8
    ):
        super().__init__(n_channels, n_heads, d_k, n_groups, False)

    def forward(self, x: Tensor, t: Tensor):
        """
        Apply attention over the spatial dimensions.

        Args:
            x (Tensor): Input tensor of shape (BxCxTxHxW).
            t (Tensor): Input tensor of shape (BxD).

        Returns:
            Tensor: Output tensor of shape (BxCxTxHxW), after applying the attention mechanism.

        """
        B = x.shape[0]

        # Reshape
        h = x
        x = rearrange(x, "b c t h w -> (b t) c h w")

        # Forward Pass
        x = super().forward(x)

        # Reshape
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B)

        return x + h


class TemporalAttention(AttentionBlock):
    def __init__(
        self,
        n_channels: int,
        n_heads: int,
        d_k: int = None,
        n_groups: int = 8,
        use_rotary_emb: bool = False,
    ):
        super().__init__(n_channels, n_heads, d_k, n_groups, use_rotary_emb)

    def forward(self, x: Tensor, t: Tensor):
        """
        Apply attention over the temporal dimension.

        Args:
            x (Tensor): Input tensor of shape (BxCxTxHxW).
            t (Tensor): Time tensor of shape (BxD).

        Returns:
            Tensor: Output tensor after applying temporal attention, with the same shape as the input tensor.
        """
        B, C, T, H, W = x.shape
        # Reshape
        h = x
        x = rearrange(x, "b c t h w -> (b h w) c t 1")

        # Forward Pass
        x = super().forward(x)

        # Reshape
        x = rearrange(x, "(b h w) c t k -> b c (t k) h w", b=B, h=H, w=W)

        return x + h


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class ConditionalUnet3DDhariwal(nn.Module):
    def __init__(
        self,
        img_channels: int,
        in_channels: int,
        time_channels: int,
        resolutions: list[int],
        n_heads: int,
        use_rotary_emb: bool,
        label_dim: int,
        label_dropout: float,
        use_context: bool,
        use_instruction: bool,
        n_agents: int = 4,
        n_frames: int = 0,
        n_context_frames: int = 0,
        context_conditioning_type: str = "time",
    ):
        super().__init__()
        self.n_context_frames = n_context_frames
        self.use_context = use_context
        self.use_instruction = use_instruction
        self.label_dropout = label_dropout
        self.context_conditioning_type = context_conditioning_type

        if self.context_conditioning_type == "channel":
            self.init_convolution = nn.Conv3d(
                (img_channels * 2) + (n_frames - 1) * n_context_frames,
                in_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            )
        elif (
            self.context_conditioning_type == "time"
            or self.context_conditioning_type == "agent_id"
            or self.context_conditioning_type == "action_space"
        ):
            self.init_convolution = nn.Conv3d(
                (img_channels * 2),
                in_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
            )
        else:
            raise NotImplementedError(
                f"This conditioning type ({self.context_conditioning_type}) is not implemented yet"
            )

        self.map_noise = PositionalEmbedding(num_channels=time_channels)
        self.map_layer0 = nn.Linear(time_channels, time_channels)
        self.map_layer1 = nn.Linear(time_channels, time_channels)
        self.map_label = nn.Linear(label_dim, time_channels)
        self.agent_embeddings = nn.Embedding(n_agents, time_channels)
        if self.context_conditioning_type == "action_space":
            self.action_space_embeddings = nn.Linear(19, time_channels)

        # Down Block
        self.down = nn.ModuleList([])

        for k, multiplier in enumerate(resolutions):
            out_channels = in_channels * multiplier

            resnet_block_1 = TimeResidualBlock(
                in_channels, out_channels, time_channels, d2=False
            )
            resnet_block_2 = TimeResidualBlock(
                out_channels, out_channels, time_channels, d2=False
            )
            spatial_attention = SpatialAttention(out_channels, n_heads=n_heads)
            temporal_attention = TemporalAttention(
                out_channels, n_heads=n_heads, use_rotary_emb=use_rotary_emb
            )

            block = nn.ModuleList(
                [
                    resnet_block_1,
                    resnet_block_2,
                    spatial_attention,
                    temporal_attention,
                ]
            )

            downsampling = (
                Downsample(out_channels, d2=False)
                if k < len(resolutions) - 1
                else nn.Identity()
            )
            self.down.append(nn.ModuleList([block, downsampling]))
            in_channels = out_channels

        # Middle Block
        self.mid_resnet_prev = TimeResidualBlock(
            out_channels, out_channels, time_channels, d2=False
        )
        self.mid_spatial_attention = SpatialAttention(out_channels, n_heads=n_heads)
        self.mid_temporal_attention = TemporalAttention(
            out_channels, n_heads=n_heads, use_rotary_emb=use_rotary_emb
        )
        self.mid_resnet_after = TimeResidualBlock(
            out_channels, out_channels, time_channels, d2=False
        )

        # Upward Block
        self.up = nn.ModuleList([])
        for k, multiplier in enumerate(reversed(resolutions)):
            out_channels = in_channels // multiplier
            resnet_block_1 = TimeResidualBlock(
                2 * in_channels, out_channels, time_channels, d2=False
            )
            resnet_block_2 = TimeResidualBlock(
                out_channels, out_channels, time_channels, d2=False
            )
            spatial_attention = SpatialAttention(out_channels, n_heads=n_heads)
            temporal_attention = TemporalAttention(out_channels, n_heads=n_heads)

            block = nn.ModuleList(
                [resnet_block_1, resnet_block_2, spatial_attention, temporal_attention]
            )

            upsample = (
                Upsample(out_channels, d2=False)
                if k < len(resolutions) - 1
                else nn.Identity()
            )
            self.up.append(nn.ModuleList([block, upsample]))
            in_channels = out_channels

        # Final convolution
        self.final_convolution = nn.Conv3d(out_channels, img_channels, 1)

    def forward(
        self, x: Tensor, noise: Tensor, obs_0: Tensor, context: Tensor, label: Tensor
    ):
        """
        x: Tensor (Video) BxTxCxHxW
        t: Tensor (Diffusion Timtesteps) Bx1
        """
        width, height = x.shape[-2], x.shape[-3]
        if width % 2 != 0:
            # Pad the images to make the size divisible by 2
            x = F.pad(x, (0, 0, 0, 1, 0, 1), mode="constant", value=0)
            obs_0 = F.pad(obs_0, (0, 0, 0, 1, 0, 1), mode="constant", value=0)
            if self.context_conditioning_type == "time":
                context = F.pad(context, (0, 0, 0, 1, 0, 1), mode="constant", value=0)

            used_padding = True
        else:
            used_padding = False

        # Prepare context
        if self.context_conditioning_type == "agent_id":
            context = self.agent_embeddings(context)

        if self.context_conditioning_type == "action_space":
            context = self.action_space_embeddings(context)

        # Apply dropout
        tmp = label.detach()
        if self.training and self.label_dropout:
            prob = torch.rand([x.shape[0]], device=x.device)
            tmp = tmp * (prob.reshape(-1, 1) >= self.label_dropout).to(label.dtype)
            obs_0 = obs_0 * (prob.reshape(-1, 1, 1, 1) >= self.label_dropout).to(
                obs_0.dtype
            )
            context = context * (prob.reshape(-1, 1, 1, 1, 1) >= self.label_dropout).to(
                context.dtype
            )

        # Map noise to embedding and add class/instruction embedding
        emb = self.map_noise(noise)
        emb = F.silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.use_instruction:
            emb = emb + self.map_label(tmp)
        emb = F.silu(emb)

        # Prepare video input
        T = x.shape[1]
        x_0 = obs_0.unsqueeze(1).repeat(1, T, 1, 1, 1)

        if self.use_context:
            if self.context_conditioning_type == "channel":
                context = (
                    rearrange(context, "b t h w c -> b h w (t c)")
                    .unsqueeze(1)
                    .repeat(1, T, 1, 1, 1)
                )
                inpt = [x, x_0, context]
                x = torch.concat(inpt, dim=-1)

            elif self.context_conditioning_type == "time":
                n_context_frames = context.shape[1]
                obs_0_context = (
                    torch.zeros_like(obs_0)
                    .unsqueeze(1)
                    .repeat(1, n_context_frames, 1, 1, 1)
                )

                x_0 = torch.cat([obs_0_context, x_0], dim=1)
                x = torch.concat([context, x], dim=1)
                inpt = [x, x_0]

            elif (
                self.context_conditioning_type == "agent_id"
                or self.context_conditioning_type == "action_space"
            ):
                emb = emb + context
                inpt = [x, x_0]

            else:
                raise NotImplementedError(
                    f"This  context conditioning type ({self.context_conditioning_type})  is not implemented yet"
                )
        else:
            inpt = [x, x_0]

        x = torch.concat(inpt, dim=-1)
        x = rearrange(x, "b t h w c -> b c t h w")

        # Apply first convolution
        x = self.init_convolution(x)

        # Go downward path
        skip = []
        for block, downsample in self.down:
            for network in block:
                x = network(x, emb)
            skip.append(x)
            x = downsample(x)

        # Middle Part
        x = self.mid_spatial_attention(x, emb)
        x = self.mid_temporal_attention(x, emb)

        # Go upward path
        assert len(skip) == len(self.up)
        for block, upsample in self.up:
            h = skip.pop()

            # Concatenate skip connection

            x = torch.cat([x, h], dim=1)

            for network in block:
                x = network(x, emb)

            x = upsample(x)

        # Final convolution
        x = self.final_convolution(x)

        if self.context_conditioning_type == "time" and self.use_context:
            x = x[:, :, n_context_frames:]

        if used_padding:
            x = x[:, :, :, :-1, :-1]

        return x
