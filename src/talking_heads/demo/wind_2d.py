import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from talking_heads.architectures.gano import create_gano
from talking_heads.loss import GaussianNLLLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gp_baseline_nll(
	pos_obs: torch.Tensor,
	x_obs: torch.Tensor,
	pos_query: torch.Tensor,
	y_true: torch.Tensor,
	criterion,
	cfg,
):
	"""Fit a standard sklearn GP per output channel and report Gaussian NLL."""
	try:
		from sklearn.gaussian_process import GaussianProcessRegressor
		from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
	except ImportError as exc:
		raise ImportError(
			"scikit-learn is required for the GP baseline. Install with: pip install scikit-learn"
		) from exc

	n_query = pos_query.shape[0]
	if cfg.gp_eval_points > 0 and cfg.gp_eval_points < n_query:
		idx = torch.randperm(n_query)[: cfg.gp_eval_points]
	else:
		idx = torch.arange(n_query)

	pos_query_eval = pos_query[idx]
	y_true_eval = y_true[idx]

	x_train = pos_obs.detach().cpu().numpy()
	y_train = x_obs.detach().cpu().numpy()
	x_test = pos_query_eval.detach().cpu().numpy()

	gp_mean = []
	gp_logvar = []

	for d in range(y_train.shape[1]):
		kernel = (
			ConstantKernel(cfg.gp_amplitude ** 2, constant_value_bounds="fixed")
			* RBF(length_scale=cfg.gp_lengthscale, length_scale_bounds="fixed")
			+ WhiteKernel(noise_level=cfg.gp_noise, noise_level_bounds="fixed")
		)

		gp = GaussianProcessRegressor(
			kernel=kernel,
			alpha=cfg.gp_alpha,
			normalize_y=True,
			n_restarts_optimizer=0,
		)

		gp.fit(x_train, y_train[:, d])
		mean_d, std_d = gp.predict(x_test, return_std=True)

		gp_mean.append(mean_d)
		gp_logvar.append(np.log(np.maximum(std_d ** 2, 1e-8)))

	gp_mean = torch.tensor(np.stack(gp_mean, axis=-1), dtype=torch.float32)
	gp_logvar = torch.tensor(np.stack(gp_logvar, axis=-1), dtype=torch.float32)

	gp_loss = criterion(gp_mean, gp_logvar, y_true_eval)
	return gp_loss.item(), idx


def generate_high_altitude_wind(
	n_grid: int,
	n_modes: int,
	base_theta_deg: float = 10.0,
	direction_jitter_deg: float = 180.0,
):
	"""Create a smooth, low-frequency vector field with a dominant flow direction."""
	x = np.linspace(0.0, 1.0, n_grid)
	y = np.linspace(0.0, 1.0, n_grid)
	X, Y = np.meshgrid(x, y)

	theta = np.deg2rad(base_theta_deg + np.random.randn() * direction_jitter_deg)
	main_dir = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
	perp_dir = np.array([-main_dir[1], main_dir[0]], dtype=np.float32)

	# Broad jet profile: stronger flow near one latitudinal band.
	jet_center = np.random.uniform(0.35, 0.65)
	jet_width = np.random.uniform(0.15, 0.28)
	jet_profile = np.exp(-0.5 * ((Y - jet_center) / jet_width) ** 2)

	# Low-frequency longitudinal and latitudinal modulation of speed.
	speed = np.random.uniform(6.0, 12.0) * np.ones_like(X)
	speed += np.random.uniform(4.0, 8.0) * jet_profile

	for _ in range(n_modes):
		kx = np.random.randint(1, 3)
		ky = np.random.randint(1, 3)
		phase = np.random.rand() * 2.0 * np.pi
		amp = np.random.uniform(0.2, 1.0) / (kx + ky)
		speed += amp * np.sin(2.0 * np.pi * (kx * X + ky * Y) + phase)

	# Weak cross-flow derived from a smooth streamfunction perturbation.
	psi = np.zeros_like(X)
	for _ in range(max(1, n_modes // 2)):
		kx = np.random.randint(1, 3)
		ky = np.random.randint(1, 3)
		phase = np.random.rand() * 2.0 * np.pi
		amp = np.random.uniform(0.05, 0.2) / (kx**2 + ky**2)
		psi += amp * np.cos(2.0 * np.pi * (kx * X + ky * Y) + phase)

	dpsi_dy, dpsi_dx = np.gradient(psi, y, x)
	cross_component = 0.6 * dpsi_dy - 0.4 * dpsi_dx

	u = speed * main_dir[0] + cross_component * perp_dir[0]
	v = speed * main_dir[1] + cross_component * perp_dir[1]

	coords = np.stack([X, Y], axis=-1)
	field = np.stack([u, v], axis=-1)
	return coords, field


def sample_observations(coords, field, n_obs: int):
	coords_flat = coords.reshape(-1, 2)
	field_flat = field.reshape(-1, 2)

	idx = np.random.choice(coords_flat.shape[0], n_obs, replace=False)
	pos_obs = coords_flat[idx]
	x_obs = field_flat[idx]
	return pos_obs, x_obs


def prepare_batch(n_grid: int, n_obs: int, n_modes: int):
	coords, field = generate_high_altitude_wind(n_grid=n_grid, n_modes=n_modes)
	pos_obs, x_obs = sample_observations(coords, field, n_obs=n_obs)

	pos_query = coords.reshape(-1, 2)
	y_true = field.reshape(-1, 2)

	return (
		torch.tensor(x_obs, dtype=torch.float32),
		torch.tensor(pos_obs, dtype=torch.float32),
		torch.tensor(pos_query, dtype=torch.float32),
		torch.tensor(y_true, dtype=torch.float32),
	)


def evaluate_and_plot(model, cfg, step=None):
	model.eval()
	x_obs, pos_obs, pos_query, y_true = prepare_batch(
		n_grid=cfg.n_grid,
		n_obs=cfg.n_obs,
		n_modes=cfg.n_modes,
	)

	with torch.no_grad():
		pred_mean, pred_logvar = model(
			x_obs=x_obs.to(device),
			pos_obs=pos_obs.to(device),
			pos_query=pos_query.to(device),
		)

	pred_mean = pred_mean.squeeze(0).cpu().numpy()
	pred_var = torch.exp(pred_logvar.squeeze(0)).cpu().numpy()
	truth = y_true.cpu().numpy()

	n = cfg.n_grid
	pred_field = pred_mean.reshape(n, n, 2)
	true_field = truth.reshape(n, n, 2)
	var_field = pred_var.reshape(n, n, 2)

	speed_true = np.linalg.norm(true_field, axis=-1)
	speed_pred = np.linalg.norm(pred_field, axis=-1)

	fig, axs = plt.subplots(2, 3, figsize=(13, 7))
	axs[0, 0].imshow(speed_true, origin="lower")
	axs[0, 0].set_title("True speed")
	axs[0, 1].imshow(speed_pred, origin="lower")
	axs[0, 1].set_title("Pred speed")
	axs[0, 2].imshow(np.linalg.norm(var_field, axis=-1), origin="lower")
	axs[0, 2].set_title("Pred variance norm")

	ds = max(1, n // 16)
	ys = np.arange(0, n, ds)
	xs = np.arange(0, n, ds)
	XX, YY = np.meshgrid(xs, ys)

	axs[1, 0].quiver(
		XX,
		YY,
		true_field[::ds, ::ds, 0],
		true_field[::ds, ::ds, 1],
		scale=70,
	)
	axs[1, 0].set_title("True vectors")

	axs[1, 1].quiver(
		XX,
		YY,
		pred_field[::ds, ::ds, 0],
		pred_field[::ds, ::ds, 1],
		scale=70,
	)
	axs[1, 1].set_title("Pred vectors")

	axs[1, 2].imshow(np.abs(speed_pred - speed_true), origin="lower")
	axs[1, 2].set_title("|speed error|")

	for ax in axs.flat:
		ax.set_xticks([])
		ax.set_yticks([])

	plt.tight_layout()

	run_dir = os.path.join(cfg.fig_dir, f"step_{step}") if step is not None else cfg.fig_dir
	os.makedirs(run_dir, exist_ok=True)
	out_name = "wind_train.png" if step is not None else "wind_final.png"
	plt.savefig(os.path.join(run_dir, out_name), dpi=150)
	plt.close(fig)


def train(cfg):
	model = create_gano(
		data_in_dim=2,
		positional_dim=2,
		data_out_dim=2,
		latent_dim=cfg.latent_dim,
		architecture="meanvar-kgnn",
		kernel='gano'
	).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
	criterion = GaussianNLLLoss()

	for step in range(cfg.steps):
		model.train()
		x_obs, pos_obs, pos_query, y_true = prepare_batch(
			n_grid=cfg.n_grid,
			n_obs=cfg.n_obs,
			n_modes=cfg.n_modes,
		)

		pred_mean, pred_logvar = model(
			x_obs=x_obs.to(device),
			pos_obs=pos_obs.to(device),
			pos_query=pos_query.to(device),
		)

		loss = criterion(pred_mean, pred_logvar, y_true.to(device))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if step % cfg.log_every == 0:
			gp_loss, eval_idx = gp_baseline_nll(
				pos_obs=pos_obs,
				x_obs=x_obs,
				pos_query=pos_query,
				y_true=y_true,
				criterion=criterion,
				cfg=cfg,
			)

			pred_mean_eval = pred_mean.squeeze(0)[eval_idx]
			pred_logvar_eval = pred_logvar.squeeze(0)[eval_idx]
			y_true_eval = y_true[eval_idx].to(device)
			no_eval_loss = criterion(pred_mean_eval, pred_logvar_eval, y_true_eval)

			print(f"Output shape: {pred_mean.shape}, {pred_logvar.shape}")
			print(
				f"Step {step:5d} | "
				f"NO(full): {loss.item():.6f} | "
				f"NO(eval): {no_eval_loss.item():.6f} | "
				f"GP(eval): {gp_loss:.6f} | "
				f"NO-GP(eval): {(no_eval_loss.item() - gp_loss):+.6f} | "
				f"n_eval={eval_idx.numel()}"
			)

		if cfg.plot_every > 0 and step % cfg.plot_every == 0:
			evaluate_and_plot(model, cfg, step=step)

	evaluate_and_plot(model, cfg, step=None)
	return model


def build_arg_parser():
	parser = argparse.ArgumentParser(
		description="Train GANO on synthetic smooth high-altitude-like wind fields."
	)
	parser.add_argument("--n-grid", type=int, default=64)
	parser.add_argument("--n-obs", type=int, default=128)
	parser.add_argument("--n-modes", type=int, default=4)
	parser.add_argument("--latent-dim", type=int, default=16)
	parser.add_argument("--steps", type=int, default=500000)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--log-every", type=int, default=1000)
	parser.add_argument("--plot-every", type=int, default=5000)
	parser.add_argument("--fig-dir", type=str, default="./.figs/wind_2d_gano")
	parser.add_argument("--seed", type=int, default=7)
	parser.add_argument("--gp-eval-points", type=int, default=4096)
	parser.add_argument("--gp-lengthscale", type=float, default=0.15)
	parser.add_argument("--gp-amplitude", type=float, default=8.0)
	parser.add_argument("--gp-noise", type=float, default=1e-2)
	parser.add_argument("--gp-alpha", type=float, default=1e-6)
	return parser


def main():
	parser = build_arg_parser()
	cfg = parser.parse_args()

	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)

	print(f"Device: {device}")
	print(f"Observation coverage: {cfg.n_obs / (cfg.n_grid * cfg.n_grid):.4%}")

	model = train(cfg)
	total_params = sum(p.numel() for p in model.parameters())
	print(f"Total parameters: {total_params:,}")
	print(f"Saved plots to: {cfg.fig_dir}")


if __name__ == "__main__":
	main()
