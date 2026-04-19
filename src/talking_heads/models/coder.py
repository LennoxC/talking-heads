import torch
import torch.nn as nn
import typing
from typing import Literal

class GANOEncoder(nn.Module):
    def __init__(
        self,
        in_dim_obs: int,
        latent_dim: int,
        activation: str = 'ReLU',
    ):
        super().__init__()

        self.obs_encoder = nn.Sequential(
            nn.Linear(in_dim_obs, latent_dim),
            getattr(nn, activation)(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(
        self,
        x_obs, # (N_o, d_in)
    ):
        # Encode observations into latent space
        z_obs = self.obs_encoder(x_obs)  # (N_o, latent_dim)

        return z_obs
    
class GANOBackgroundEncoder(nn.Module):
    def __init__(
        self,
        bg_dim: int,
        latent_dim: int,
        activation: str = 'ReLU',
    ):
        super().__init__()

        self.bg_encoder = nn.Sequential(
            nn.Linear(bg_dim, latent_dim),
            getattr(nn, activation)(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(
        self,
        x_bg, # (d_b,)
    ):
        # Encode background information into latent space
        z_bg = self.bg_encoder(x_bg)  # (latent_dim,)

        return z_bg
    
class GANOMeanDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        bg_dim: typing.Optional[int] = None,
        activation: str = 'ReLU'
    ):
        super().__init__()

        decoder_in = latent_dim * 2 if bg_dim is not None else latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, latent_dim),
            getattr(nn, activation)(),
            nn.Linear(latent_dim, out_dim) # output mean only
        )
    
    def forward(self, x):
        x = self.decoder(x)
        
        return x

class GANOMeanVarDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        bg_dim: typing.Optional[int] = None,
        activation: str = 'ReLU'
    ):
        super().__init__()

        decoder_in = latent_dim * 2 if bg_dim is not None else latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, latent_dim),
            getattr(nn, activation)(),
            nn.Linear(latent_dim, 2 * out_dim) # output mean and logvar
        )
    
    def forward(self, x):
        x = self.decoder(x)
        
        return torch.chunk(x, 2, dim=-1) # mean, logvar