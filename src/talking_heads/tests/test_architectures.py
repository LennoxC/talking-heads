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
    {
        "name": "meanvar-gnn-activations",
        "kwargs": dict(
            data_in_dim=4,
            positional_dim=2,
            data_out_dim=3,
            architecture="meanvar-kgnn",
            activations={
                'encoder': 'GELU',
                'bg_encoder': 'LeakyReLU',
                'gnn': 'Sigmoid',
                'kernel': 'Tanh',
                'decoder': 'SiLU'
            })
    }
]

@pytest.fixture(params=MODEL_CONFIGS, ids=lambda x: x["name"])
def model_and_config(request):
    config = request.param
    model = create_gano(**config["kwargs"])
    return model, config

def make_dummy_batch(config, n_obs=200, n_query=64):
    """
    Create a batched dataset with variable number of observations per sample.
    """

    kwargs = config["kwargs"]

    data_in_dim = kwargs["data_in_dim"]
    positional_dim = kwargs["positional_dim"]

    B = 4  # batch size

    obs_pos_list = []
    obs_data_list = []
    obs_batch_list = []
    obs_mask_list = []

    query_pos_list = []
    query_batch_list = []

    total_obs = 0
    total_query = 0

    for b in range(B):
        # Vary number of observations per batch element
        n_obs_b = torch.randint(low=50, high=150, size=(1,)).item()
        n_query_b = n_query  # keep grid size fixed (can also vary if desired)

        obs_pos_b = torch.randn(n_obs_b, positional_dim)
        obs_data_b = torch.randn(n_obs_b, data_in_dim)

        query_pos_b = torch.randn(n_query_b, positional_dim)

        # Masks (all valid for now, but structure is important)
        obs_mask_b = torch.ones(n_obs_b, dtype=torch.bool)

        # Batch indices
        obs_batch_b = torch.full((n_obs_b,), b, dtype=torch.long)
        query_batch_b = torch.full((n_query_b,), b, dtype=torch.long)

        obs_pos_list.append(obs_pos_b)
        obs_data_list.append(obs_data_b)
        obs_mask_list.append(obs_mask_b)
        obs_batch_list.append(obs_batch_b)

        query_pos_list.append(query_pos_b)
        query_batch_list.append(query_batch_b)

        total_obs += n_obs_b
        total_query += n_query_b

    # Concatenate into flat tensors (PyG-style batching)
    obs_pos = torch.cat(obs_pos_list, dim=0)
    obs_data = torch.cat(obs_data_list, dim=0)
    obs_mask = torch.cat(obs_mask_list, dim=0)
    obs_batch = torch.cat(obs_batch_list, dim=0)

    query_pos = torch.cat(query_pos_list, dim=0)
    query_batch = torch.cat(query_batch_list, dim=0)

    return obs_pos, obs_data, query_pos, obs_mask, obs_batch, query_batch

def run_model(model, obs_pos, obs_data, query_pos, obs_mask, obs_batch, query_batch):
    out = model(
        x_obs=obs_data,
        pos_obs=obs_pos,
        pos_query=query_pos,
        #obs_mask=obs_mask,
        obs_batch=obs_batch,
        query_batch=query_batch,
    )

    if isinstance(out, tuple):
        mean, logvar = out
    else:
        mean, logvar = out, None
        
    return mean, logvar

def test_forward_pass(model_and_config):
    model, config = model_and_config
    out_dim = config["kwargs"]["data_out_dim"]
    
    obs_pos, obs_data, query_pos, obs_mask, obs_batch, query_batch = make_dummy_batch(config)
     
    mean, logvar = run_model(model, obs_pos, obs_data, query_pos, obs_mask, obs_batch, query_batch)
     
    assert mean.shape == (query_pos.shape[0], out_dim)
    
    if logvar is not None:
        assert logvar is not None
        assert logvar.shape == mean.shape

def test_backward_pass(model_and_config):
    model, config = model_and_config
    
    obs_pos, obs_data, query_pos, obs_mask, obs_batch, query_batch = make_dummy_batch(config)
     
    mean, logvar = run_model(model, obs_pos, obs_data, query_pos, obs_mask, obs_batch, query_batch)
    
    loss = GaussianNLLLoss()(mean, logvar, torch.randn_like(mean)) if logvar is not None else L2Loss()(mean, torch.randn_like(mean))
    loss.backward()
    
    grads_exist = [
        p.grad is not None for p in model.parameters() if p.requires_grad
    ]
    
    assert any(grads_exist)

def test_no_nans(model_and_config):
    model, config = model_and_config
    
    obs_pos, obs_data, query_pos, obs_mask, obs_batch, query_batch = make_dummy_batch(config)
    
    mean, logvar = run_model(model, obs_pos, obs_data, query_pos, obs_mask, obs_batch, query_batch)

    assert torch.isfinite(mean).all()
    if logvar is not None:
        assert torch.isfinite(logvar).all()

def test_batch_isolation(model_and_config):
    model, config = model_and_config
    
    obs_pos, obs_data, query_pos, obs_mask, obs_batch, query_batch = make_dummy_batch(config)

    mean, _ = run_model(model, obs_pos, obs_data, query_pos, obs_mask, obs_batch, query_batch)

    # Pick two different batch elements
    b0 = 0
    b1 = 1

    q0 = (query_batch == b0)
    q1 = (query_batch == b1)

    # Ensure outputs are not identical across batches (sanity check)
    assert not torch.allclose(mean[q0], mean[q1])

