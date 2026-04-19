import pytest
import torch
from talking_heads.models.gnn import GNN

CONFIGS = [
    {
        "name": "knn-gnn-noloops",
        "kwargs": dict(
            latent_dim=16,
            arch='k',
            k=4,
            layers=2,
            self_loops=False,
        )
    },
    {
        "name": "radius-gnn-loops",
        "kwargs": dict(
            latent_dim=16,
            arch='r',
            r=1.0,
            layers=2,
            self_loops=True,
        )
    },
    {
        "name": "knn-gnn-3layers-highdim",
        "kwargs": dict(
            latent_dim=32,
            arch='k',
            k=4,
            layers=3,
            self_loops=True,
        )
    },
    {
        "name": "knn-gnn-GELU",
        "kwargs": dict(
            latent_dim=16,
            arch='k',
            k=4,
            layers=2,
            self_loops=True,
            activation='GELU'
        )
    }
]

@pytest.fixture(params=CONFIGS, ids=lambda x: x["name"])
def gnn_and_config(request):
    config = request.param
    gnn = GNN(**config["kwargs"])
    return gnn, config

def sample_data(n_obs=100, latent_dim=16, pos_dim=2):
    h_obs = torch.randn(n_obs, latent_dim)
    pos_obs = torch.randn(n_obs, pos_dim)
    return h_obs, pos_obs

@pytest.mark.parametrize("pos_dim", [2, 3])
def test_gnn_forward(gnn_and_config, pos_dim):
    gnn, config = gnn_and_config
    kwargs = config["kwargs"]
    h_obs, pos_obs = sample_data(n_obs=50, latent_dim=kwargs["latent_dim"], pos_dim=pos_dim)
    h_bg = torch.randn_like(h_obs)  # Background features (not used in current GNN implementation but included for compatibility)
    
    output = gnn(h_obs, h_bg, pos_obs)
    
    assert output.shape == h_obs.shape, f"Expected output shape {h_obs.shape}, got {output.shape}"

@pytest.mark.parametrize("pos_dim", [2, 3])
def test_gnn_backward(gnn_and_config, pos_dim):
    gnn, config = gnn_and_config
    kwargs = config["kwargs"]
    h_obs, pos_obs = sample_data(n_obs=50, latent_dim=kwargs["latent_dim"], pos_dim=pos_dim)
    h_bg = torch.randn_like(h_obs)
    
    output = gnn(h_obs, h_bg, pos_obs)
    loss = output.sum() # Simple loss to test backward pass
    loss.backward()  # Check that backward pass works without error