import talking_heads
import torch
import torch.nn as nn
from .coder import GANOEncoder, GANOBackgroundEncoder, GANOMeanDecoder, GANOMeanVarDecoder
from .gano_kernel import GANOKernel
from .gnn import GNN
from typing import Optional, Literal, List

class GraphAttentionNeuralOperator(nn.Module):
    def __init__(
        self,
        in_dim_obs,
        pos_dim,
        latent_dim,
        out_dim,
        batch_size=1,
        heads=16,
        radius=None,
        output_mode: Literal['MeanVar', 'Mean'] = 'MeanVar',
        distance_encoding: List[Literal['q_pos', 'o_pos', 'rel', 'dist']] = ['q_pos', 'o_pos', 'rel', 'dist'], # TODO: support for rbf and fourier encodings
        gnn_arch: Literal['r', 'k'] = 'k',
        gnn_layers: int = 2,
        gnn_k: int = 4,
        gnn_r: float = 1.0,
        gnn_self_loops: bool = True,
        activations: dict = {'encoder': 'ReLU', 'bg_encoder': 'ReLU', 'gnn': 'ReLU', 'kernel': 'ReLU', 'decoder': 'ReLU'}
    ):
        super().__init__()
        self.proj_dim = latent_dim * heads # latent_dim is the latent dim per head. The total dim therefore needs to be multiplied by the number of heads.

        # ---- Observation encoder ----
        # Projects observations into latent space.
        # allows choice of activation function.
        self.obs_encoder = GANOEncoder(
            in_dim_obs=in_dim_obs,
            latent_dim=self.proj_dim,
            activation=activations['encoder']
        )

        # ---- Graph Neural Network ----
        # Multi-layer message passing system. This gives an opportunity 
        # to learn a more complex kernel. Local features can be aggregated 
        # rather than relying on single node representations to learn the kernel.
        # Form the GNN only knowing the latent dim of the node features.
        # The GNN implements KNN/Radius graph construction internally.
        self.gnn = GNN(
            latent_dim=self.proj_dim,
            arch=gnn_arch,
            k=gnn_k,
            r=gnn_r,
            layers=gnn_layers,
            self_loops=gnn_self_loops,
            activation=activations['gnn']
        )

        # ---- Kernel & Attention ----
        self.kernel = GANOKernel(
            pos_dim=pos_dim,
            latent_dim=self.proj_dim,
            out_dim=out_dim,
            radius=radius,
            heads=heads,
            activation=activations['kernel'],
            distance_encoding=distance_encoding
        )

        # ---- Output: mean + var ----
        if output_mode not in ['MeanVar', 'Mean']: raise ValueError(f"Invalid output_mode: {output_mode}. Must be 'MeanVar' or 'Mean'.")

        self.decoder = getattr(talking_heads.models.coder, f"GANO{output_mode}Decoder")(
            latent_dim=self.proj_dim,
            out_dim=out_dim,
            activation=activations['decoder']
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
        obs_batch=None, # (N_o,)
        query_batch=None # (N_q,)
    ):
        # ---- Encode obs ----
        h_obs = self.obs_encoder(x_obs)  # (N_o, d)

        # ---- GNN message passing ----
        h_obs = self.gnn(
            h_obs=h_obs,
            pos_obs=pos_obs,
            batch=obs_batch
        ) # (N_o, d)

        # if batch dim is none, reshape to add a single batch dim
        if obs_batch is None:
            h_obs = h_obs.unsqueeze(0)
            pos_obs = pos_obs.unsqueeze(0)
            pos_query = pos_query.unsqueeze(0)

        h_query = self.kernel(
            h_obs=h_obs,
            pos_obs=pos_obs,
            pos_query=pos_query,
            obs_batch=obs_batch,
            query_batch=query_batch
        ) # (N_q, d)

        # ---- Decode & Return ----
        return self.decoder(h_query) # (N_q, 2*out_dim) -> mean + logvar