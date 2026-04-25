import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import knn_graph, radius_graph
from typing import Literal, List

class GNN(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            arch: Literal['r', 'k'] = 'k',
            k: int = 4,
            r: float = 1.0,
            layers: int = 2,
            self_loops: bool = True,
            activation: str = 'ReLU',
        ):
        super().__init__()

        self.k = k
        self.r = r
        self.arch = arch
        if arch == 'r':
            self.graph_constructor = lambda pos, batch: radius_graph(pos, r=self.r, batch=batch, loop=self_loops)
        else: # default to knn (arch = 'k')
            self.graph_constructor = lambda pos, batch: knn_graph(pos, k=self.k, batch=batch, loop=self_loops)
        self.convs = nn.ModuleList([
            # TODO: specify more params in the sage conv (normalization, aggregator etc)
            SAGEConv(latent_dim, latent_dim) for _ in range(layers)
        ])
        self.activation = getattr(nn, activation)()

    def forward(
        self,
        h_obs,  # (N_o, d_h)
        pos_obs, # (N_o, d_p)
        batch=None # (N_o,)
    ):
        # Construct graph using KNN
    
        if batch is None:
            batch = pos_obs.new_zeros(pos_obs.shape[0], dtype=torch.long)

        edge_index = self.graph_constructor(pos_obs, batch) # (2, E)
        
        # Message passing
        h = h_obs
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.activation(h)

        # output must be (N_o, d_h) to be compatible with the kernel.
        return h