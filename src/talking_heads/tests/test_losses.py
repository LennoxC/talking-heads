import pytest
import torch
from talking_heads.loss import GaussianNLLLoss, L2Loss


def test_l2loss_precomputed_value():
	loss_fn = L2Loss()

	pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
	target = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

	# Squared errors: [0, 1, 1, 4] -> mean = 1.5
	expected = torch.tensor(1.5)

	loss_mean = loss_fn(pred, target)
	loss_sum = loss_fn(pred, target)

	assert torch.isclose(loss_mean, expected)
	assert torch.isclose(loss_sum, expected)


def test_gaussian_nll_precomputed_value_zero_logvar():
	loss_fn = GaussianNLLLoss()

	mean = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
	target = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
	logvar = torch.zeros_like(mean)

	# For logvar = 0, nll reduces to mean squared error term in this implementation.
	# Squared errors: [0, 1, 1, 4] -> mean = 1.5
	expected = torch.tensor(1.5)

	loss_mean = loss_fn(mean, logvar, target)
	loss_sum = loss_fn(mean, logvar, target)

	assert torch.isclose(loss_mean, expected)
	assert torch.isclose(loss_sum, expected)


def test_l2loss_random_output_properties():
    torch.manual_seed(0)
    loss_fn = L2Loss()

    pred = torch.randn(16, 8)
    target = torch.randn(16, 8)
    loss = loss_fn(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss >= 0


def test_gaussian_nll_random_output_properties():
    torch.manual_seed(1)
    loss_fn = GaussianNLLLoss()

    mean = torch.randn(32, 4)
    target = torch.randn(32, 4)
    logvar = torch.randn(32, 4).clamp(min=-5.0, max=5.0)

    loss = loss_fn(mean, logvar, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

def test_gaussian_nll_is_zero_when_prediction_matches_target_and_logvar_zero():
	loss_fn = GaussianNLLLoss()

	target = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
	mean = target.clone()
	logvar = torch.zeros_like(target)

	loss = loss_fn(mean, logvar, target)
	assert torch.isclose(loss, torch.tensor(0.0))


def test_gaussian_nll_increases_with_larger_error_when_logvar_fixed():
	loss_fn = GaussianNLLLoss()

	target = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
	logvar = torch.zeros_like(target)

	mean_small_error = torch.tensor([[0.1, -0.1], [0.0, 0.0]])
	mean_large_error = torch.tensor([[1.0, -1.0], [0.0, 0.0]])

	small_loss = loss_fn(mean_small_error, logvar, target)
	large_loss = loss_fn(mean_large_error, logvar, target)

	assert large_loss > small_loss
