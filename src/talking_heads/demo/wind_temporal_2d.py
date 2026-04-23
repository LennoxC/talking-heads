import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from talking_heads.architectures.gano import create_gano
from talking_heads.loss import GaussianNLLLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gp_baseline_nll(
	pos_obs_xy: torch.Tensor,
	x_obs: torch.Tensor,
	pos_query_xy: torch.Tensor,
	y_true: torch.Tensor,
	criterion,
	cfg,
):
	"""Fit a deliberately misspecified GP baseline that ignores time in the coordinates."""
	try:
		from sklearn.gaussian_process import GaussianProcessRegressor
		from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
	except ImportError as exc:
		raise ImportError(
			"scikit-learn is required for the GP baseline. Install with: pip install scikit-learn"
		) from exc

	n_query = pos_query_xy.shape[0]
	if cfg.gp_eval_points > 0 and cfg.gp_eval_points < n_query:
		idx = torch.randperm(n_query)[: cfg.gp_eval_points]
	else:
		idx = torch.arange(n_query)

	query_xy_eval = pos_query_xy[idx]
	y_true_eval = y_true[idx]

	x_train = pos_obs_xy.detach().cpu().numpy()
	y_train = x_obs.detach().cpu().numpy()
	x_test = query_xy_eval.detach().cpu().numpy()

	gp_mean = []
	gp_logvar = []

	for d in range(y_train.shape[1]):
		kernel = (
			ConstantKernel(cfg.gp_amplitude**2, constant_value_bounds="fixed")
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
		gp_logvar.append(np.log(np.maximum(std_d**2, 1e-8)))

	gp_mean = torch.tensor(np.stack(gp_mean, axis=-1), dtype=torch.float32)
	gp_logvar = torch.tensor(np.stack(gp_logvar, axis=-1), dtype=torch.float32)

	gp_loss = criterion(gp_mean, gp_logvar, y_true_eval)
	return gp_loss.item(), idx


def generate_temporal_high_altitude_wind(
	n_grid: int,
	n_times: int,
	n_modes: int,
	base_theta_deg: float = 18.0,
	direction_jitter_deg: float = 70.0,
	temporal_speed_drift: float = 0.06,
	temporal_dir_drift_deg: float = 6.0,
):
	"""Create smooth high-altitude-like winds with gentle time drift.

	Returns:
		coords: (T, H, W, 3) containing x, y, t
		field: (T, H, W, 2) containing u, v
		t_values: (T,) time coordinates in [-1, 0]
	"""
	x = np.linspace(0.0, 1.0, n_grid)
	y = np.linspace(0.0, 1.0, n_grid)
	t_values = np.linspace(-1.0, 0.0, n_times)
	X, Y = np.meshgrid(x, y)

	theta0 = np.deg2rad(base_theta_deg + np.random.randn() * direction_jitter_deg)
	base_main_dir = np.array([np.cos(theta0), np.sin(theta0)], dtype=np.float32)

	jet_center = np.random.uniform(0.35, 0.65)
	jet_width = np.random.uniform(0.12, 0.25)
	jet_profile = np.exp(-0.5 * ((Y - jet_center) / jet_width) ** 2)

	base_speed = np.random.uniform(6.0, 11.0) * np.ones_like(X)
	base_speed += np.random.uniform(3.0, 7.0) * jet_profile

	for _ in range(n_modes):
		kx = np.random.randint(1, 4)
		ky = np.random.randint(1, 4)
		phase = np.random.rand() * 2.0 * np.pi
		amp = np.random.uniform(0.2, 1.0) / (kx + ky)
		base_speed += amp * np.sin(2.0 * np.pi * (kx * X + ky * Y) + phase)

	psi_base = np.zeros_like(X)
	for _ in range(max(1, n_modes // 2)):
		kx = np.random.randint(1, 3)
		ky = np.random.randint(1, 3)
		phase = np.random.rand() * 2.0 * np.pi
		amp = np.random.uniform(0.04, 0.16) / (kx**2 + ky**2)
		psi_base += amp * np.cos(2.0 * np.pi * (kx * X + ky * Y) + phase)

	coords = np.zeros((n_times, n_grid, n_grid, 3), dtype=np.float32)
	field = np.zeros((n_times, n_grid, n_grid, 2), dtype=np.float32)

	for i, t in enumerate(t_values):
		time_phase = 2.0 * np.pi * (t + 1.0)

		theta_t = theta0 + np.deg2rad(temporal_dir_drift_deg) * np.sin(0.8 * time_phase)
		main_dir_t = np.array([np.cos(theta_t), np.sin(theta_t)], dtype=np.float32)
		perp_dir_t = np.array([-main_dir_t[1], main_dir_t[0]], dtype=np.float32)

		speed_t = base_speed * (1.0 + temporal_speed_drift * np.sin(time_phase))
		speed_t += 0.45 * temporal_speed_drift * np.cos(2.0 * np.pi * (X - 0.7 * Y) + 1.3 * time_phase)

		psi_t = psi_base * (1.0 + 0.55 * temporal_speed_drift * np.cos(1.1 * time_phase))
		dpsi_dy, dpsi_dx = np.gradient(psi_t, y, x)
		cross_component = 0.65 * dpsi_dy - 0.35 * dpsi_dx

		u = speed_t * main_dir_t[0] + cross_component * perp_dir_t[0]
		v = speed_t * main_dir_t[1] + cross_component * perp_dir_t[1]

		coords[i, :, :, 0] = X
		coords[i, :, :, 1] = Y
		coords[i, :, :, 2] = t

		field[i, :, :, 0] = u.astype(np.float32)
		field[i, :, :, 1] = v.astype(np.float32)

	return coords, field, t_values.astype(np.float32)


def sample_observations(coords, field, n_obs: int):
	"""Sample space-time observations uniformly over the full temporal cube."""
	flat_coords = coords.reshape(-1, 3)
	flat_field = field.reshape(-1, 2)

	idx = np.random.choice(flat_coords.shape[0], n_obs, replace=False)
	pos_obs = flat_coords[idx]
	x_obs = flat_field[idx]
	return pos_obs, x_obs


def prepare_batch(n_grid: int, n_times: int, n_obs: int, n_modes: int):
	coords, field, t_values = generate_temporal_high_altitude_wind(
		n_grid=n_grid,
		n_times=n_times,
		n_modes=n_modes,
	)

	pos_obs, x_obs = sample_observations(coords, field, n_obs=n_obs)

	# Query only final time slice (t = 0) for every spatial grid point.
	coords_final = coords[-1]
	field_final = field[-1]

	pos_query = coords_final.reshape(-1, 3)
	y_true = field_final.reshape(-1, 2)

	return (
		torch.tensor(x_obs, dtype=torch.float32),
		torch.tensor(pos_obs, dtype=torch.float32),
		torch.tensor(pos_query, dtype=torch.float32),
		torch.tensor(y_true, dtype=torch.float32),
		torch.tensor(t_values, dtype=torch.float32),
	)


def evaluate_and_plot(model, cfg, step=None):
	model.eval()
	x_obs, pos_obs, pos_query, y_true, t_values = prepare_batch(
		n_grid=cfg.n_grid,
		n_times=cfg.n_times,
		n_obs=cfg.n_obs,
		n_modes=cfg.n_modes,
	)

	criterion = GaussianNLLLoss()

	with torch.no_grad():
		pred_mean, pred_logvar = model(
			x_obs=x_obs.to(device),
			pos_obs=pos_obs.to(device),
			pos_query=pos_query.to(device),
		)

	pred_mean = pred_mean.squeeze(0).cpu()
	pred_logvar = pred_logvar.squeeze(0).cpu()
	pred_var = torch.exp(pred_logvar)

	no_final_nll = criterion(pred_mean, pred_logvar, y_true).item()

	gp_final_nll, eval_idx = gp_baseline_nll(
		pos_obs_xy=pos_obs[:, :2],
		x_obs=x_obs,
		pos_query_xy=pos_query[:, :2],
		y_true=y_true,
		criterion=criterion,
		cfg=cfg,
	)

	no_eval_nll = criterion(pred_mean[eval_idx], pred_logvar[eval_idx], y_true[eval_idx]).item()

	n = cfg.n_grid
	true_field = y_true.numpy().reshape(n, n, 2)
	pred_field = pred_mean.numpy().reshape(n, n, 2)
	var_field = pred_var.numpy().reshape(n, n, 2)

	speed_true = np.linalg.norm(true_field, axis=-1)
	speed_pred = np.linalg.norm(pred_field, axis=-1)
	speed_err = np.abs(speed_pred - speed_true)

	u_true = true_field[:, :, 0]
	v_true = true_field[:, :, 1]
	u_pred = pred_field[:, :, 0]
	v_pred = pred_field[:, :, 1]
	u_var = var_field[:, :, 0]
	v_var = var_field[:, :, 1]

	fig, axs = plt.subplots(3, 4, figsize=(16, 10))

	panels = [
		(speed_true, "True speed"),
		(speed_pred, "Pred speed"),
		(speed_err, "|speed error|"),
		(np.sqrt(u_var + v_var), "Uncertainty norm"),
		(u_true, "True u"),
		(u_pred, "Pred u"),
		(np.abs(u_pred - u_true), "|u error|"),
		(u_var, "Var u"),
		(v_true, "True v"),
		(v_pred, "Pred v"),
		(np.abs(v_pred - v_true), "|v error|"),
		(v_var, "Var v"),
	]

	for ax, (img, title) in zip(axs.flat, panels):
		im = ax.imshow(img, origin="lower")
		ax.set_title(title)
		ax.set_xticks([])
		ax.set_yticks([])
		fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

	plt.suptitle(
		f"Final-time diagnostics (t=0) | NO NLL={no_final_nll:.4f} | GP(eval)={gp_final_nll:.4f}",
		y=1.02,
	)
	plt.tight_layout()

	run_dir = os.path.join(cfg.fig_dir, f"step_{step}") if step is not None else cfg.fig_dir
	os.makedirs(run_dir, exist_ok=True)
	out_name = "wind_temporal_train_maps.png" if step is not None else "wind_temporal_final_maps.png"
	plt.savefig(os.path.join(run_dir, out_name), dpi=160, bbox_inches="tight")
	plt.close(fig)

	fig2, axs2 = plt.subplots(2, 2, figsize=(13, 9))

	axs2[0, 0].hist(pos_obs[:, 2].numpy(), bins=max(8, cfg.n_times), alpha=0.85, color="#2b8cbe")
	axs2[0, 0].set_title("Observation time distribution")
	axs2[0, 0].set_xlabel("t")
	axs2[0, 0].set_ylabel("count")

	sc = axs2[0, 1].scatter(
		pos_obs[:, 0].numpy(),
		pos_obs[:, 1].numpy(),
		c=pos_obs[:, 2].numpy(),
		s=8,
		alpha=0.75,
		cmap="viridis",
	)
	axs2[0, 1].set_title("Observed (x, y) colored by time")
	axs2[0, 1].set_xlabel("x")
	axs2[0, 1].set_ylabel("y")
	fig2.colorbar(sc, ax=axs2[0, 1], fraction=0.046, pad=0.03)

	speed_true_flat = speed_true.reshape(-1)
	speed_pred_flat = speed_pred.reshape(-1)
	axs2[1, 0].scatter(speed_true_flat, speed_pred_flat, s=9, alpha=0.35, color="#d95f0e")
	lo = min(speed_true_flat.min(), speed_pred_flat.min())
	hi = max(speed_true_flat.max(), speed_pred_flat.max())
	axs2[1, 0].plot([lo, hi], [lo, hi], "k--", linewidth=1.2)
	axs2[1, 0].set_title("Speed parity at final time")
	axs2[1, 0].set_xlabel("true speed")
	axs2[1, 0].set_ylabel("pred speed")

	std_u = (u_pred - u_true) / np.sqrt(np.maximum(u_var, 1e-8))
	std_v = (v_pred - v_true) / np.sqrt(np.maximum(v_var, 1e-8))
	axs2[1, 1].hist(std_u.reshape(-1), bins=40, alpha=0.6, label="u", color="#8856a7")
	axs2[1, 1].hist(std_v.reshape(-1), bins=40, alpha=0.6, label="v", color="#31a354")
	axs2[1, 1].set_title("Standardized residuals")
	axs2[1, 1].set_xlabel("(pred - true) / sigma")
	axs2[1, 1].set_ylabel("count")
	axs2[1, 1].legend()

	plt.tight_layout()
	out_name2 = "wind_temporal_train_stats.png" if step is not None else "wind_temporal_final_stats.png"
	plt.savefig(os.path.join(run_dir, out_name2), dpi=160, bbox_inches="tight")
	plt.close(fig2)

	metrics = {
		"no_final_nll": no_final_nll,
		"no_eval_nll": no_eval_nll,
		"gp_eval_nll": gp_final_nll,
		"no_minus_gp_eval": no_eval_nll - gp_final_nll,
		"n_eval": int(eval_idx.numel()),
		"obs_time_min": float(pos_obs[:, 2].min().item()),
		"obs_time_max": float(pos_obs[:, 2].max().item()),
		"query_time": float(pos_query[0, 2].item()),
		"t_values_min": float(t_values.min().item()),
		"t_values_max": float(t_values.max().item()),
	}
	return metrics


def plot_training_history(history, cfg):
	if len(history["step"]) == 0:
		return

	fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))

	steps = np.array(history["step"])
	no_eval = np.array(history["no_eval_nll"])
	gp_eval = np.array(history["gp_eval_nll"])
	delta = np.array(history["no_minus_gp_eval"])

	axs[0].plot(steps, no_eval, label="NO(eval NLL)", linewidth=2.0)
	axs[0].plot(steps, gp_eval, label="GP(eval NLL, no-time)", linewidth=2.0)
	axs[0].set_title("NLL comparison over training")
	axs[0].set_xlabel("step")
	axs[0].set_ylabel("NLL")
	axs[0].legend()
	axs[0].grid(alpha=0.25)

	axs[1].axhline(0.0, linestyle="--", linewidth=1.2, color="black")
	axs[1].plot(steps, delta, color="#d95f02", linewidth=2.0)
	axs[1].set_title("NO(eval) - GP(eval)")
	axs[1].set_xlabel("step")
	axs[1].set_ylabel("NLL delta")
	axs[1].grid(alpha=0.25)

	plt.tight_layout()
	os.makedirs(cfg.fig_dir, exist_ok=True)
	plt.savefig(os.path.join(cfg.fig_dir, "wind_temporal_training_history.png"), dpi=160)
	plt.close(fig)


def train(cfg):
	model = create_gano(
		data_in_dim=2,
		positional_dim=3,
		data_out_dim=2,
		latent_dim=cfg.latent_dim,
		architecture="meanvar-kgnn",
		kernel="gano",
	).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
	criterion = GaussianNLLLoss()

	history = {
		"step": [],
		"no_eval_nll": [],
		"gp_eval_nll": [],
		"no_minus_gp_eval": [],
	}

	for step in range(cfg.steps):
		model.train()
		x_obs, pos_obs, pos_query, y_true, _ = prepare_batch(
			n_grid=cfg.n_grid,
			n_times=cfg.n_times,
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
			metrics = evaluate_and_plot(model, cfg, step=step if cfg.plot_every > 0 else None)
			history["step"].append(step)
			history["no_eval_nll"].append(metrics["no_eval_nll"])
			history["gp_eval_nll"].append(metrics["gp_eval_nll"])
			history["no_minus_gp_eval"].append(metrics["no_minus_gp_eval"])

			print(
				f"Step {step:6d} | "
				f"NO(full): {loss.item():.6f} | "
				f"NO(eval): {metrics['no_eval_nll']:.6f} | "
				f"GP(eval,xy-only): {metrics['gp_eval_nll']:.6f} | "
				f"NO-GP(eval): {metrics['no_minus_gp_eval']:+.6f} | "
				f"n_eval={metrics['n_eval']}"
			)
			print(
				f"Time coverage obs=[{metrics['obs_time_min']:.3f}, {metrics['obs_time_max']:.3f}] "
				f"query_t={metrics['query_time']:.3f} grid_t=[{metrics['t_values_min']:.3f}, {metrics['t_values_max']:.3f}]"
			)

		if cfg.plot_every > 0 and step % cfg.plot_every == 0 and step % cfg.log_every != 0:
			evaluate_and_plot(model, cfg, step=step)

	final_metrics = evaluate_and_plot(model, cfg, step=None)
	history["step"].append(cfg.steps)
	history["no_eval_nll"].append(final_metrics["no_eval_nll"])
	history["gp_eval_nll"].append(final_metrics["gp_eval_nll"])
	history["no_minus_gp_eval"].append(final_metrics["no_minus_gp_eval"])

	plot_training_history(history, cfg)
	return model, final_metrics


def build_arg_parser():
	parser = argparse.ArgumentParser(
		description=(
			"Train GANO on synthetic high-altitude winds with mild temporal drift. "
			"Observations are sampled across (x, y, t), and predictions are made only at final time t=0."
		)
	)
	parser.add_argument("--n-grid", type=int, default=64)
	parser.add_argument("--n-times", type=int, default=8)
	parser.add_argument("--n-obs", type=int, default=256)
	parser.add_argument("--n-modes", type=int, default=4)
	parser.add_argument("--latent-dim", type=int, default=16)
	parser.add_argument("--steps", type=int, default=500000)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--log-every", type=int, default=500)
	parser.add_argument("--plot-every", type=int, default=2000)
	parser.add_argument("--fig-dir", type=str, default="./.figs/wind_temporal_2d_gano")
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

	max_obs = cfg.n_grid * cfg.n_grid * cfg.n_times
	if cfg.n_obs > max_obs:
		raise ValueError(f"n_obs={cfg.n_obs} exceeds available space-time points ({max_obs}).")

	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)

	print(f"Device: {device}")
	print(f"Space-time observation coverage: {cfg.n_obs / max_obs:.4%}")
	print("Model uses coordinates (x, y, t); GP baseline intentionally uses only (x, y).")

	model, final_metrics = train(cfg)
	total_params = sum(p.numel() for p in model.parameters())

	print(f"Total parameters: {total_params:,}")
	print(
		f"Final metrics | NO(eval): {final_metrics['no_eval_nll']:.6f} | "
		f"GP(eval): {final_metrics['gp_eval_nll']:.6f} | "
		f"NO-GP(eval): {final_metrics['no_minus_gp_eval']:+.6f}"
	)
	print(f"Saved diagnostics to: {cfg.fig_dir}")


if __name__ == "__main__":
	main()
