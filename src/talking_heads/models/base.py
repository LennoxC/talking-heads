import torch
import torch.nn as nn
from .coder import GANOEncoder, GANOBackgroundEncoder, GANODecoder
from .kernel import GANOKernel
from .gnn import GNN
from typing import Optional, Literal, List

# TODO: Allow more parameters for customization of the model.
# This might be best achieved through a create_gano function.
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

        # ---- Graph Neural Network ----
        # Multi-layer message passing system. This gives an opportunity 
        # to learn a more complex kernel. Local features can be aggregated 
        # rather than relying on single node representations to learn the kernel.
        # Form the GNN only knowing the latent dim of the node features.
        # The GNN implements KNN/Radius graph construction internally.
        self.gnn = GNN(
            latent_dim=latent_dim,
            k=4,
            layers=2,
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

        # ---- GNN message passing ----
        if self.use_bg:
            h_bg = self.bg_encoder(x_bg) # (d,)
        else:
            h_bg = None
        
        h_obs = self.gnn(
            h_obs=h_obs,
            h_bg=h_bg,
            pos_obs=pos_obs
        ) # (N_o, d)

        h_query = self.kernel(
            h_obs=h_obs,
            pos_obs=pos_obs,
            pos_query=pos_query,
            h_bg=h_bg,
            obs_mask=obs_mask
        ) # (N_q, d)

        # ---- Decode & Return ----
        return self.decoder(h_query) # (N_q, 2*out_dim) -> mean + logvar