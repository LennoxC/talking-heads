import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import knn_graph

class GNN(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            k: int = 4,
            layers: int = 2,
            activation: str = 'ReLU',
        ):
        super().__init__()

        self.k = k
        self.convs = nn.ModuleList([
            SAGEConv(latent_dim, latent_dim) for _ in range(layers)
        ])
        self.activation = getattr(nn, activation)()

    def forward(
        self,
        h_obs,  # (N_o, d_h)
        h_bg,   # (N_o, d_h)
        pos_obs # (N_o, d_p)
    ):
        # Construct graph using KNN
    
        # x = [N_o, d_h]
        edge_index = knn_graph(pos_obs, k=self.k, batch=None, loop=True) # (2, E)
        
        # Message passing
        h = h_obs
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.activation(h)

        # output must be (N_o, d_h) to be compatible with the kernel.
        return h