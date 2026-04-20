import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from talking_heads.models.base import GraphAttentionNeuralOperator
from talking_heads.loss import GaussianNLLLoss, L2Loss
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_MODES = 20
N_GRID = 64
N_OBS = 200

def generate_field(n_grid=N_GRID, n_modes=N_MODES):
    x = np.linspace(0, 1, n_grid)
    y = np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(x, y)

    psi = np.zeros_like(X)

    # random Fourier-like smooth field
    for _ in range(n_modes):
        kx, ky = np.random.randint(1, 5, size=2)
        phase = np.random.rand() * 2 * np.pi
        amp = np.random.randn() * 0.5

        psi += amp * np.sin(2 * np.pi * (kx * X + ky * Y) + phase)

    # velocity from streamfunction
    u = np.gradient(psi, axis=0)
    v = -np.gradient(psi, axis=1)

    # scalar field (correlated but not identical)
    # scalar = psi + 0.1 * np.random.randn(*psi.shape)

    field = np.stack([u, v], axis=-1)  # (H, W, 2)

    coords = np.stack([X, Y], axis=-1)  # (H, W, 2)

    return coords, field

def generate_field_with_bg(n_grid=N_GRID, n_modes=N_MODES):
    x = np.linspace(0, 1, n_grid)
    y = np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(x, y)

    psi = np.zeros_like(X)

    # random Fourier-like smooth field
    for _ in range(n_modes):
        kx, ky = np.random.randint(1, 5, size=2)
        phase = np.random.rand() * 2 * np.pi
        amp = np.random.randn() * 0.5

        psi += amp * np.sin(2 * np.pi * (kx * X + ky * Y) + phase)

    # velocity from streamfunction
    u = np.gradient(psi, axis=0)
    v = -np.gradient(psi, axis=1)

    # scalar field (correlated but not identical)
    #scalar = psi + 0.1 * np.random.randn(*psi.shape)

    field = np.stack([u, v], axis=-1)  # (H, W, 2)
    #bg = scalar.unsqueeze(-1) # (H, W, 1)
    #bg = np.expand_dims(scalar, axis=-1)

    coords = np.stack([X, Y], axis=-1)  # (H, W, 2)

    return coords, field#, bg

def sample_observations(coords, field, n_obs=N_OBS):
    H, W, _ = coords.shape
    coords_flat = coords.reshape(-1, 2)
    field_flat = field.reshape(-1, 2)

    idx = np.random.choice(len(coords_flat), n_obs, replace=False)

    pos_obs = coords_flat[idx]
    x_obs = field_flat[idx]

    return pos_obs, x_obs

def sample_observations_with_bg(coords, field, bg, n_obs=N_OBS):
    H, W, _ = coords.shape
    coords_flat = coords.reshape(-1, 2)
    field_flat = field.reshape(-1, 2)
    bg_flat = bg.reshape(-1, 1)

    idx = np.random.choice(len(coords_flat), n_obs, replace=False)

    pos_obs = coords_flat[idx]
    x_obs = field_flat[idx]
    x_bg = bg_flat[idx]

    return pos_obs, x_obs, x_bg

def prepare_batch(n_grid=N_GRID, n_obs=N_OBS):
    coords, field = generate_field(n_grid)

    pos_obs, x_obs = sample_observations(coords, field, n_obs)

    pos_query = coords.reshape(-1, 2)
    y_true = field.reshape(-1, 2)

    return (
        torch.tensor(x_obs, dtype=torch.float32),
        torch.tensor(pos_obs, dtype=torch.float32),
        torch.tensor(pos_query, dtype=torch.float32),
        torch.tensor(y_true, dtype=torch.float32),
    )

def prepare_batch_with_bg(n_grid=N_GRID, n_obs=N_OBS):
    coords, field, bg = generate_field_with_bg(n_grid)

    pos_obs, x_obs, x_bg = sample_observations_with_bg(coords, field, bg, n_obs)

    pos_query = coords.reshape(-1, 2)
    y_true = field.reshape(-1, 2)
    x_bg = bg.reshape(-1, 1)

    return (
        torch.tensor(x_obs, dtype=torch.float32),
        torch.tensor(pos_obs, dtype=torch.float32),
        torch.tensor(pos_query, dtype=torch.float32),
        torch.tensor(y_true, dtype=torch.float32),
        torch.tensor(x_bg, dtype=torch.float32)
    )

'''
model = GraphAttentionNeuralOperator(
    in_dim_obs=2, # u and v in
    pos_dim=2, # x and y coordinates
    latent_dim=64, # dimension of latent node features
    out_dim=2, # u and v out
    bg_dim=None, # scalar background field dimension
    radius=0.3, # local attention radius (in normalized coordinates)
    output_mode='MeanVar', # output both mean and variance for uncertainty estimation
).to(device)
'''

from talking_heads.architectures.gano import create_gano

model = create_gano(
    data_in_dim=2,
    positional_dim=2,
    data_out_dim=2,
    latent_dim=16
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_step():
    model.train()

    x_obs, pos_obs, pos_query, y_true = prepare_batch()

    x_obs = x_obs.to(device)  # Use only the first two dimensions
    pos_obs = pos_obs.to(device)
    pos_query = pos_query.to(device)
    y_true = y_true.to(device)

    pred_mean, pred_logvar = model(
        x_obs=x_obs,
        pos_obs=pos_obs,
        pos_query=pos_query
    )

    # Gaussian NLL loss
    loss = GaussianNLLLoss()(pred_mean, pred_logvar, y_true)
    # loss = ((pred_mean - y_true) ** 2 * torch.exp(-pred_logvar) + pred_logvar).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def evaluate_and_plot(step=None):
    model.eval()

    x_obs, pos_obs, pos_query, y_true = prepare_batch(n_grid=N_GRID, n_obs=N_OBS)

    with torch.no_grad():
        pred_mean, pred_var = model(
            x_obs=x_obs.to(device),
            pos_obs=pos_obs.to(device),
            pos_query=pos_query.to(device)
        )

    pred = pred_mean.cpu().numpy()
    var = (torch.exp(pred_var)).cpu().numpy() # convert logvar to var
    truth = y_true.numpy()

    n = int(np.sqrt(len(pred)))

    pred = pred.reshape(n, n, 2)
    truth = truth.reshape(n, n, 2)
    var = var.reshape(n, n, 2)

    fig, axs = plt.subplots(3, 2, figsize=(12, 6))

    titles = ["u", "v"]

    for i in range(2):
        axs[0, i].imshow(truth[:, :, i])
        axs[0, i].set_title(f"True {titles[i]}")

        axs[1, i].imshow(pred[:, :, i])
        axs[1, i].set_title(f"Pred {titles[i]}")

        axs[2, i].imshow(var[:, :, i])
        axs[2, i].set_title(f"Variance {titles[i]}")

    # make a directory for this training step
    if step is not None:
        os.makedirs(f"./.figs/graph_interp/training/step_{step}", exist_ok=True)

    path = f"./.figs/graph_interp/training/step_{step}/graph_interp.png" if step is not None else "./.figs/graph_interp/training/graph_interp_final.png"

    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    # now plot two subplots: the true and predicted quiver plots of the velocity field
    """
    plt.figure(figsize=(6, 6))
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))

    plt.subplot(2, 1, 1)
    plt.quiver(truth[:, :, 0], truth[:, :, 1], color='blue', alpha=0.5, label='True')
    plt.title("Velocity Field Quiver Plot")

    plt.subplot(2, 1, 2)
    plt.quiver(pred[:, :, 0], pred[:, :, 1], color='red', alpha=0.5, label='Pred')
    plt.title("Velocity Field Quiver Plot")
    
    plt.savefig(f"./.figs/graph_interp/training/step_{step}/quiver_plot.png" if step is not None else "./.figs/graph_interp/training/quiver_plot_final.png")
    plt.close()
    """

def run():
    for step in range(10000):
        loss = train_step()
        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")
            evaluate_and_plot(step=step)

if __name__ == "__main__":
    print("Sample coverage: ", N_OBS / (N_GRID * N_GRID), "%")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    run()
    evaluate_and_plot()