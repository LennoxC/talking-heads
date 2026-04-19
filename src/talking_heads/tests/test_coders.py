import pytest
import talking_heads
import torch
from talking_heads.models.coder import GANOEncoder, GANOMeanDecoder, GANOMeanVarDecoder


@pytest.mark.parametrize("latent_dim, activation", [
    (16, 'ReLU'),
    (32, 'GELU'),
    (16, 'Sigmoid'),
])
def test_encoder_forward(latent_dim, activation):
    torch.manual_seed(0)
    in_dim_obs = 8
    encoder = GANOEncoder(in_dim_obs=in_dim_obs, latent_dim=latent_dim, activation=activation)

    x_obs = torch.randn(10, in_dim_obs)  # 10 observations
    z_obs = encoder(x_obs)

    assert z_obs.shape == (10, latent_dim)
    assert torch.isfinite(z_obs).all()

@pytest.mark.parametrize("latent_dim, activation", [
    (16, 'ReLU'),
    (32, 'GELU'),
    (16, 'Sigmoid'),
])
def test_encoder_backward(latent_dim, activation):
    torch.manual_seed(0)
    in_dim_obs = 8
    encoder = GANOEncoder(in_dim_obs=in_dim_obs, latent_dim=latent_dim, activation=activation)

    x_obs = torch.randn(10, in_dim_obs)  # 10 observations
    z_obs = encoder(x_obs)

    loss = z_obs.sum()
    loss.backward()

    for param in encoder.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()

@pytest.mark.parametrize("decoder, latent_dim, out_dim, activation", [
    ('GANOMeanDecoder', 16, 2, 'ReLU'),
    ('GANOMeanDecoder', 32, 3, 'GELU'),
    ('GANOMeanVarDecoder', 16, 2, 'ReLU'),
    ('GANOMeanVarDecoder', 32, 3, 'Tanh'),
])
def test_decoder_forward(decoder, latent_dim, out_dim, activation):
    torch.manual_seed(0)
    decoder = getattr(talking_heads.models.coder, decoder)(latent_dim=latent_dim, out_dim=out_dim, activation=activation)

    z = torch.randn(10, latent_dim)  # 10 latent vectors

    out = decoder(z)

    if isinstance(out, tuple):
        mean, logvar = out
        assert mean.shape == (10, out_dim)
        assert logvar.shape == (10, out_dim)
        assert torch.isfinite(mean).all()
        assert torch.isfinite(logvar).all()
    else:
        mean = out
        assert mean.shape == (10, out_dim)
        assert torch.isfinite(mean).all()