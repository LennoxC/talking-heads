import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralGPKernel(nn.Module):
    def __init__(
        self,
        in_dim_obs,
        pos_dim,
        latent_dim,
        out_dim,
        bg_dim=None,
        heads=4,
        head_dim=None,
        activation='ReLU',
        learn_sigma=True
    ):
        super().__init__()

        self.use_bg = bg_dim is not None

        self.heads = heads
        self.head_dim = head_dim or (latent_dim // heads)
        self.latent_dim = self.heads * self.head_dim

        act = getattr(nn, activation)

        # --- Feature encoder (like GP feature map) ---
        self.feature_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            act(),
            nn.Linear(latent_dim, self.latent_dim)
        )

        # --- Learnable length scales (multi-head = multi-scale GP) ---
        if learn_sigma:
            self.log_sigma = nn.Parameter(torch.zeros(heads))
        else:
            self.register_buffer("log_sigma", torch.zeros(heads))

        # --- Optional learned kernel correction ---
        self.kernel_mlp = nn.Sequential(
            nn.Linear(pos_dim * 3, latent_dim),
            act(),
            nn.Linear(latent_dim, heads)
        )

        # --- Output projection ---
        self.out_proj = nn.Linear(self.latent_dim + (bg_dim or 0), out_dim)

        # --- Variance head (GP-like uncertainty) ---
        self.var_proj = nn.Linear(self.latent_dim, out_dim)

    def forward(
        self,
        h_obs,        # (B, N_o, d_h)
        pos_obs,      # (B, N_o, d_p)
        pos_query,    # (B, N_q, d_p)
        h_bg=None,    # (B, N_q, d_bg)
        obs_mask=None # (B, N_o)
    ):
        B, N_o, _ = h_obs.shape
        _, N_q, _ = pos_query.shape

        # --- Encode observations ---
        v = self.feature_net(h_obs)  # (B, N_o, H*d_head)
        v = v.view(B, N_o, self.heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)  # (B, H, N_o, d_head)

        # --- Pairwise distances ---
        pos_q = pos_query.unsqueeze(2)  # (B, N_q, 1, d_p)
        pos_o = pos_obs.unsqueeze(1)    # (B, 1, N_o, d_p)

        rel = pos_q - pos_o             # (B, N_q, N_o, d_p)
        dist2 = (rel ** 2).sum(dim=-1)  # (B, N_q, N_o)

        # --- RBF kernel (GP core) ---
        sigma = torch.exp(self.log_sigma).view(1, self.heads, 1, 1)
        rbf = torch.exp(-dist2.unsqueeze(1) / (sigma ** 2 + 1e-6))  # (B, H, N_q, N_o)

        # --- Learned kernel correction ---
        pos_q_exp = pos_q.expand(-1, -1, N_o, -1)
        pos_o_exp = pos_o.expand(-1, N_q, -1, -1)

        kernel_input = torch.cat([pos_q_exp, pos_o_exp, rel], dim=-1)
        delta = self.kernel_mlp(kernel_input)  # (B, N_q, N_o, H)
        delta = delta.permute(0, 3, 1, 2)

        logits = torch.log(rbf + 1e-8) + delta  # combine GP prior + learned correction

        # --- Masking ---
        if obs_mask is not None:
            logits = logits.masked_fill(
                obs_mask.unsqueeze(1).unsqueeze(2) == 0,
                float('-inf')
            )

        # --- Normalized weights (CRITICAL: kriging-style) ---
        weights = torch.softmax(logits, dim=-1)  # (B, H, N_q, N_o)

        # --- Mean prediction ---
        h_query = torch.matmul(weights, v)  # (B, H, N_q, d_head)
        h_query = h_query.permute(0, 2, 1, 3).contiguous()
        h_query = h_query.view(B, N_q, self.latent_dim)

        # --- GP-style variance (distance-weighted uncertainty) ---
        # variance ≈ weighted squared residuals
        v_mean = torch.matmul(weights, v)  # reuse
        residual = v.unsqueeze(2) - v_mean.unsqueeze(3)  # (B, H, N_q, N_o, d_head)
        var = (weights.unsqueeze(-1) * residual**2).sum(dim=3)
        var = var.permute(0, 2, 1, 3).reshape(B, N_q, self.latent_dim)

        # --- Background fusion ---
        if self.use_bg:
            h_query = torch.cat([h_query, h_bg], dim=-1)

        # --- Final outputs ---
        mean = self.out_proj(h_query)
        var = F.softplus(self.var_proj(var))  # ensure positive

        return mean, var