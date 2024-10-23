import random
import time

from einops import rearrange
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch import Tensor
from torch.optim import Adam
from tqdm import tqdm


def get_variance_schedule(
    timesteps: int, schedule_type: str, beta_start: float = None, beta_end: float = None
):
    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, timesteps)
    elif schedule_type == "cosine":
        f = (
            lambda t: torch.cos(
                torch.tensor(
                    (((t / timesteps) + 0.008) / 1.008) * torch.pi * 0.5,
                    dtype=torch.float,
                )
            )
            ** 2
        )
        alpha_prod = torch.tensor(
            [f(t) / f(0) for t in range(timesteps + 1)], dtype=torch.float
        )
        betas = 1 - (alpha_prod[1:] / alpha_prod[:-1])
        return torch.clip(betas, 0, 0.999)


class BaseDiffusion(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        T: int,
        variance_schedule: str,
        beta_start: float,
        beta_end: float,
        lr: float,
        mean: float,
        std: float,
        l1: bool,
        use_instruction: bool,
        use_context: bool,
        conditional_prob: float,
        cond_w: float,
        model_type: str
    ):
        """
        T: number of diffusion steps
        beta_start: starting value of beta
        beta_end: value of beta at last diffusion step
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.T = T
        self.model_type = model_type
        self.lr = lr
        self.betas = get_variance_schedule(
            self.T, variance_schedule, beta_start, beta_end
        )
        self._betas2alphas()
        self.l1 = l1

        if self.l1:
            self.loss = nn.L1Loss(reduce="mean")
        else:
            self.loss = nn.MSELoss(reduce="mean")

        self.model = model

        # Arguments related to training a conditional diffusion model
        self.use_instruction = use_instruction
        self.use_context = use_context
        self.conditional_prob = conditional_prob
        self.register_buffer("cond_w", torch.tensor(cond_w, dtype=torch.float32))

        self.transform = False

    def _betas2alphas(self):
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))

    def forward_diffusion(self, x0, error, timesteps):
        B = x0.shape[0]
        shape = (1,) * (len(x0.shape) - 1)
        alphas = self.alpha_bars[timesteps].reshape(B, *shape)
        xt = torch.sqrt(alphas) * x0 + torch.sqrt(1 - alphas) * error
        return xt

    def backward_diffusion(self, obs_0, context, label, shape, sampling_timesteps, sampling_strategy):
        B = obs_0.shape[0]
        xt = torch.randn(size=(B, *shape)).to(self.device)
        xts = [xt]

        label = label.to(self.device)
        obs_0 = obs_0.to(self.device)
        context = context.to(self.device)

        for t,t_1 in tqdm(zip(reversed(sampling_timesteps[1:]),reversed(sampling_timesteps[:-1]))):
            xt = self.backward_step(xt, t, t_1, obs_0, context, label, shape, sampling_strategy)
            if self.transform:
                xt = self.transform(xt, reverse=True)
            xts.append(xt)

        return xts

    def backward_step(self, xt, t, t_1, obs_0, context, label, shape, sampling_strategy):
        B = xt.shape[0]

        if isinstance(t,int):
            t = torch.tensor(t).long().to(xt.device).repeat(B)
        if isinstance(t_1,int):
            t_1 = torch.tensor(t_1).long().to(xt.device).repeat(B)

        latent = torch.randn(size=(B, *shape), device=self.device)

        if not self.use_instruction:
            label = None
            
        if not self.use_context:
            context = torch.zeros_like(context)

        conditional_model_output = self.model(
            xt, t, obs_0, context, label
        )

  
        if self.cond_w > 0:
            
            label = None 
            context = torch.zeros_like(context)
            obs_0 = torch.zeros_like(obs_0)

            unconditional_model_output = self.model(
                xt,
                t,
                obs_0,
                context,
                label,
            )
            model_output = (1+self.cond_w)*conditional_model_output - self.cond_w*unconditional_model_output 

        else:
            model_output = conditional_model_output

        model_output = rearrange(model_output, "b c t h w -> b t h w c")

        alpha_bar_t = self.alpha_bars[t].reshape(-1, 1, 1, 1, 1)
        mask = t_1 == 0
        alpha_bar_t_minus_1 = self.alpha_bars[t_1].reshape(-1, 1, 1, 1, 1)
        alpha_bar_t_minus_1[mask] = 1 

        if sampling_strategy == "ddpm":
            variance = ((1-alpha_bar_t_minus_1)/(1-alpha_bar_t))*(1-alpha_bar_t/alpha_bar_t_minus_1)
        elif sampling_strategy == "ddim":
            variance = torch.tensor(0).type_as(xt)
        else: 
            raise NotImplementedError(f"This sampling stratgey ({sampling_strategy}) does not exist.")

        xt = self.calculate_xt(
            xt,
            model_output,
            alpha_bar_t,
            alpha_bar_t_minus_1,
            variance,
            latent
        )

        return xt 

    def calculate_xt(self, xt, model_output, alpha_bar_t, alpha_bar_t_minus_1, variance, latent):
        
        if self.model_type=="error":
            scale = torch.sqrt(1 - alpha_bar_t)

            mean = xt - scale * model_output

            xt = (
                torch.sqrt(alpha_bar_t_minus_1) / torch.sqrt(alpha_bar_t) * mean
                + torch.sqrt(1 - alpha_bar_t_minus_1 - variance) * model_output
                + torch.sqrt(variance) * latent
            )

        elif self.model_type=="x":
            error = (1 / torch.sqrt(1 - alpha_bar_t)) * (
                xt - torch.sqrt(alpha_bar_t) * model_output
            )

            xt = (
                torch.sqrt(alpha_bar_t_minus_1) * model_output
                + torch.sqrt(1 - alpha_bar_t_minus_1 - variance) * error
                + torch.sqrt(variance) * latent
            )

        xt = xt.clamp(-1, 1)

        return xt

    def training_step(self, traj, mask, context, labels, B):
        """
        traj: batch of data
        B: batch size
        """
      
        # Separate the first observation and the rest
        obs_0, x = traj[:, 0], traj[:, 1:]

        # For each sample, sample a timestep
        timesteps = torch.randint(0, self.T, (B,)).type_as(traj).long()

        # Sample errors
        err = torch.randn_like(x)

        # Compute forward part
        xt = self.forward_diffusion(x, err, timesteps)

        # Determine conditioning information
        p = random.random()
        if self.conditional_prob < p or not self.use_instruction:
            labels = None

        if self.conditional_prob < p:
            obs_0 = torch.zeros_like(obs_0)

        if self.conditional_prob < p or not self.use_context:
            context = torch.zeros_like(context)

        predicted_err = self.model(xt, timesteps, obs_0, context, labels)
        predicted_err = rearrange(predicted_err, "b c t h w -> b t h w c")

        # Padded error
        mask = mask[:, 1:]
        padded_err = torch.masked_select(err, mask=mask.bool())
        padded_predicted_err = torch.masked_select(predicted_err, mask=mask.bool())

        loss = self.loss(padded_err, padded_predicted_err)
        self.log("training/loss", loss)

        return loss

    def validation_step(self, traj, mask, context, labels, B):
        # Separate the first observation and the rest
        obs_0, x = traj[:, 0], traj[:, 1:]

        # For each sample, sample a timestep
        timesteps = torch.randint(0, self.T, (B,)).type_as(traj).long()

        # Sample errors
        err = torch.randn_like(x)

        # Compute forward part
        xt = self.forward_diffusion(x, err, timesteps)

        p = random.random()
        if self.conditional_prob < p or not self.use_instruction:
            labels = None

        if self.conditional_prob < p:
            obs_0 = torch.zeros_like(obs_0)

        if self.conditional_prob < p or not self.use_context:
            context = torch.zeros_like(context)

        predicted_err = self.model(xt, timesteps, obs_0, context, labels)
        predicted_err = rearrange(predicted_err, "b c t h w -> b t h w c")

        # Padded error
        mask = mask[:, 1:]
        padded_err = torch.masked_select(err, mask=mask.bool())
        padded_predicted_err = torch.masked_select(predicted_err, mask=mask.bool())

        loss = self.loss(padded_err, padded_predicted_err)
        self.log("validation/loss", loss)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)

    def conditional_sample(self, obs_0, context, label, shape, sampling_timesteps, sampling_strategy):
        with torch.no_grad():
            xts = self.backward_diffusion(obs_0, context, label, shape, sampling_timesteps, sampling_strategy)
        return xts

    def transform(self, x, reverse=False):
        if reverse:
            return x * self.std + self.mean
        else:
            return (x - self.mean) / self.std



# Taken from: https://github.com/NVlabs/edm

class EDM(pl.LightningModule):

    def __init__(self,model, lr , P_mean, P_std, sigma_data,num_steps,min_sigma,max_sigma,rho, mean, std):

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
        loss = self.get_loss(video,context,labels)
        self.log("training/loss", loss)
        return loss 
    
    def get_loss(self,video,context,labels):
        obs_0, video = video[:, 0], video[:, 1:]

        rnd_normal = torch.randn([video.shape[0], 1, 1, 1, 1], device=video.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
    
        n = torch.randn_like(video) * sigma
        
        D_video_n = self.get_prediction(video + n, sigma, obs_0, context, labels)
        loss = weight * ((D_video_n - video) ** 2)
        
        return loss.mean()
    
    def get_prediction(self, x, sigma, obs_0, context, labels):
        sigma = sigma.reshape(-1, 1, 1, 1, 1)
        
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4


        F_x = self.model(c_in * x, c_noise.flatten(), obs_0, context, labels) 
        F_x = rearrange(F_x,"b c t h w -> b t h w c")
        
   
        D_x = c_skip * x + c_out * F_x

        return D_x
    
    def validation_step(self, video, context, labels):
        loss = self.get_loss(video,context,labels)
        self.log("validation/loss", loss)
        return loss 

    def conditional_sample(self, obs_0, context, labels, shape):

        # Send to device 
        obs_0 = obs_0.to(self.device)
        context = context.to(self.device)
        labels = labels.to(self.device) 

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, device=self.device)
        t_steps = (self.max_sigma ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (self.min_sigma ** (1 / self.rho) - self.max_sigma ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([t_steps,torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.
        B = obs_0.shape[0]
        latents = torch.randn(B,*shape).to(self.device)
        x_next = latents * t_steps[0]
       
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            # Disabled stochastic sampling
            # gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            # t_hat = self.model.round_sigma(t_cur + gamma * t_cur)

            x_hat = x_cur # + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
            t_hat = t_cur

            # Euler step.
            denoised = self.get_prediction(x_hat, t_hat.reshape(-1), obs_0, context, labels)
            
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                denoised = self.get_prediction(x_next, t_next.reshape(-1), obs_0, context, labels)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)


        return x_next

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)
        
