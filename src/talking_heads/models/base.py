import torch
import torch.nn as nn
from .coder import GANOEncoder, GANOBackgroundEncoder, GANODecoder
from .kernel import GANOKernel
from typing import Optional, Literal, List

class GraphAttentionNeuralOperator(nn.Module):
    def __init__(
        self,
        in_dim_obs,
        pos_dim,
        latent_dim,
        out_dim,
        bg_dim=None,
        radius=None,
        distance_encoding: List[Literal['q_pos', 'o_pos', 'rel', 'rbf', 'fourier']] = ['q_pos', 'o_pos', 'rel']
    ):
        super().__init__()

        self.radius = radius

        # ---- Observation encoder ----
        # Projects observations into latent space.
        # allows choice of activation function.
        self.obs_encoder = GANOEncoder(
            in_dim_obs=in_dim_obs,
            latent_dim=latent_dim,
            activation='ReLU'
        )

        # ---- Background encoder ----
        # Projects background information into latent space
        # given less non-linearities and params which might not be desirable.
        # The background information is probably equally important as the observations.
        self.use_bg = bg_dim is not None
        if self.use_bg:
            self.bg_encoder = GANOBackgroundEncoder(
                bg_dim=bg_dim,
                latent_dim=latent_dim
            )

        # ---- Kernel & Attention ----
        self.kernel = GANOKernel(
            in_dim_obs=latent_dim,
            pos_dim=pos_dim,
            latent_dim=latent_dim,
            out_dim=latent_dim,
            bg_dim=bg_dim,
            radius=radius,
            distance_encoding=distance_encoding
        )

        # ---- Output: mean + var ----
        self.decoder = GANODecoder(
            latent_dim=latent_dim,
            out_dim=out_dim,
            bg_dim=bg_dim
        )

    # N_o = number of observations (i.e. number of graph nodes)
    # N_q = number of query points (i.e. number of query points in the output field)
    # d_in = dimension of observation features (depends on number of inputs. E.g. for fluid flow, it could be velocity + pressure = 4)
    # d_p = dimension of position features (3 for 3D)
    # d_b = dimension of background features (3 for 3D)

    def forward(
        self,
        x_obs,        # (N_o, d_in)
        pos_obs,      # (N_o, d_p)
        pos_query,    # (N_q, d_p)
        x_bg=None,    # (N_q, d_b)
        obs_mask=None # (N_o,)
    ):
        # ---- Encode obs ----
        h_obs = self.obs_encoder(x_obs)  # (N_o, d)

        h_query = self.kernel(
            h_obs=h_obs,
            x_obs=x_obs,
            pos_obs=pos_obs,
            pos_query=pos_query,
            x_bg=x_bg,
            obs_mask=obs_mask
        ) # (N_q, d)

        # ---- Decode & Return ----
        return self.decoder(h_query) # (N_q, 2*out_dim) -> mean + logvar