import torch
import torch.nn as nn

class DeepONetKernel(nn.Module):
    def __init__(
        self,
        in_dim_obs,
        pos_dim,
        latent_dim,
        out_dim,
        bg_dim=None,
        heads=4,
        head_dim=None,
        activation='ReLU'
    ):
        super().__init__()

        self.use_bg = bg_dim is not None

        self.heads = heads
        self.head_dim = head_dim or (latent_dim // heads)
        self.latent_dim = self.heads * self.head_dim

        act = getattr(nn, activation)

        # --- Branch net (observations → latent function encoding) ---
        self.branch_net = nn.Sequential(
            nn.Linear(latent_dim + pos_dim, latent_dim),
            act(),
            nn.Linear(latent_dim, self.latent_dim)
        )

        # --- Trunk net (query location → basis functions) ---
        self.trunk_net = nn.Sequential(
            nn.Linear(pos_dim, latent_dim),
            act(),
            nn.Linear(latent_dim, self.latent_dim)
        )

        # Learned observation weighting for set aggregation.
        self.obs_weight = nn.Linear(self.latent_dim, 1)

        # --- Output projection ---
        self.out_proj = nn.Linear(self.latent_dim + (bg_dim or 0), out_dim)

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

        # --- Branch encoding (set → latent) ---
        branch_in = torch.cat([h_obs, pos_obs], dim=-1)  # (B, N_o, d_h + d_p)
        branch_raw = self.branch_net(branch_in)              # (B, N_o, H*d_head)

        if obs_mask is not None:
            branch_raw = branch_raw * obs_mask.unsqueeze(-1)

        # Aggregate over observations (mean pooling)
        #denom = obs_mask.sum(dim=1, keepdim=True) + 1e-6 if obs_mask is not None else N_o
        #branch = branch_raw.sum(dim=1) / denom              # (B, H*d_head)

        # weighted learning instead of mean pooling
        weights = torch.softmax(
            self.obs_weight(branch_raw), dim=1
        )
        branch = (weights * branch_raw).sum(dim=1)

        # Reshape for heads
        branch = branch.view(B, self.heads, self.head_dim)  # (B, H, d_head)

        # --- Trunk encoding (query-wise) ---
        trunk = self.trunk_net(pos_query)              # (B, N_q, H*d_head)
        trunk = trunk.view(B, N_q, self.heads, self.head_dim)

        # --- Inner product (DeepONet core) ---
        # (B, N_q, H, d_head) · (B, H, d_head) → (B, N_q, H)
        out = (trunk * branch.unsqueeze(1)).sum(dim=-1)

        # Expand back to full latent space
        out = out.unsqueeze(-1) * trunk  # (B, N_q, H, d_head)
        out = out.reshape(B, N_q, self.latent_dim)

        # --- Background fusion ---
        if self.use_bg:
            out = torch.cat([out, h_bg], dim=-1)

        # --- Final projection ---
        #out = self.out_proj(out)

        return out