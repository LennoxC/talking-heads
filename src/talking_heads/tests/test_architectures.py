import pytest
import torch

from talking_heads.architectures.gano import create_gano
from talking_heads.loss import GaussianNLLLoss, L2Loss

MODEL_CONFIGS = [
    {
        "name": "meanvar-kgnn",
        "kwargs": dict(
            data_in_dim=4,
            positional_dim=2,
            data_out_dim=3,
            architecture="meanvar-kgnn",
            gnn_layers=2,
            gnn_k=4,
        ),
    },
    {
        "name": "mean-rgnn",
        "kwargs": dict(
            data_in_dim=4,
            positional_dim=2,
            data_out_dim=3,
            architecture="mean-rgnn",
            gnn_layers=3,
            gnn_r=0.5,
        ),
    },
    {
        "name": "meanvar-no-gnn",
        "kwargs": dict(
            data_in_dim=4,
            positional_dim=2,
            data_out_dim=3,
            architecture="meanvar",
        ),
    },
]

@pytest.fixture(params=MODEL_CONFIGS, ids=lambda x: x["name"])
def model_and_config(request):
    config = request.param
    model = create_gano(**config["kwargs"])
    return model, config

def make_dummy_batch(config, n_obs=200, n_query=64):
    kwargs = config["kwargs"]
    
    data_in_dim = kwargs["data_in_dim"]
    positional_dim = kwargs["positional_dim"]
    
    obs_pos = torch.randn(n_obs, positional_dim)
    obs_data = torch.randn(n_obs, data_in_dim)
    
    query_pos = torch.randn(n_query, positional_dim)
    
    return obs_pos, obs_data, query_pos

def run_model(model, obs_pos, obs_data, query_pos):
    out = model(x_obs=obs_data, pos_obs=obs_pos, pos_query=query_pos)
    
    if isinstance(out, tuple):
        mean, logvar = out
    else:
        mean, logvar = out, None
        
    return mean, logvar

def test_forward_pass(model_and_config):
    model, config = model_and_config
    out_dim = config["kwargs"]["data_out_dim"]
    
    obs_pos, obs_data, query_pos = make_dummy_batch(config)
    
    mean, logvar = run_model(model, obs_pos, obs_data, query_pos)
    
    assert mean.shape == (query_pos.shape[0], out_dim)
    
    if logvar is not None:
        assert logvar is not None
        assert logvar.shape == mean.shape

def test_backward_pass(model_and_config):
    model, config = model_and_config
    
    obs_pos, obs_data, query_pos = make_dummy_batch(config)
    
    mean, logvar = run_model(model, obs_pos, obs_data, query_pos)
    
    loss = GaussianNLLLoss()(mean, logvar, torch.randn_like(mean)) if logvar is not None else L2Loss()(mean, torch.randn_like(mean))
    loss.backward()
    
    grads_exist = [
        p.grad is not None for p in model.parameters() if p.requires_grad
    ]
    
    assert any(grads_exist)

def test_no_nans(model_and_config):
    model, config = model_and_config
    
    obs_pos, obs_data, query_pos = make_dummy_batch(config)
    
    mean, logvar = run_model(model, obs_pos, obs_data, query_pos)

    assert torch.isfinite(mean).all()
    if logvar is not None:
        assert torch.isfinite(logvar).all()

