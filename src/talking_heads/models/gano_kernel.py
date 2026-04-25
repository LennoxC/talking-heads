import torch
import torch.nn as nn

class GANOKernel(nn.Module):
    def __init__(
        self,
        in_dim_obs,
        pos_dim,
        latent_dim,
        out_dim,
        radius=None,
        heads=4,
        activation='ReLU',
        distance_encoding=['q_pos', 'o_pos', 'rel', 'dist']
    ):
        super().__init__()

        self.radius = radius
        self.distance_encoding = distance_encoding
        self.position_encodings = len(distance_encoding)
        self.kernel_in_dim = self.position_encodings * pos_dim if 'dist' not in distance_encoding else self.position_encodings * pos_dim - pos_dim + 1

        self.heads = heads
        self.head_dim = latent_dim // heads
        self.latent_dim = self.heads * self.head_dim

        # --- Kernel MLP (shared across heads) ---
        self.kernel_mlp = nn.Sequential(
            nn.Linear(self.kernel_in_dim, latent_dim),
            getattr(nn, activation)(),
            nn.Linear(latent_dim, heads)  # one logit per head
        )

        # --- Pre-LN + value projection (multi-head) ---
        self.pre_ln = nn.LayerNorm(latent_dim)
        self.value_proj = nn.Linear(latent_dim, self.latent_dim)

        # --- Output projection ---
        self.out_proj = nn.Linear(self.latent_dim + (bg_dim or 0), out_dim)

    def forward(
        self,
        h_obs,        # (B, N_o, d_h)
        pos_obs,      # (B, N_o, d_p)
        pos_query,    # (B, N_q, d_p)
        obs_mask=None # (B, N_o)
    ):
        B, N_q, _ = pos_query.shape
        _, N_o, _ = pos_obs.shape

        # --- Pairwise geometry ---
        pos_q = pos_query.unsqueeze(2)   # (B, N_q, 1, d_p)
        pos_o = pos_obs.unsqueeze(1)     # (B, 1, N_o, d_p)

        rel = pos_q - pos_o              # (B, N_q, N_o, d_p)

        pos_q_exp = pos_q.expand(-1, -1, N_o, -1)
        pos_o_exp = pos_o.expand(-1, N_q, -1, -1)

        dist = torch.norm(rel, dim=-1, keepdim=True)

        # construct kernel input based on distance encoding choices
        kernel_input_parts = []
        if 'q_pos' in self.distance_encoding:
            kernel_input_parts.append(pos_q_exp)
        if 'o_pos' in self.distance_encoding:
            kernel_input_parts.append(pos_o_exp)
        if 'rel' in self.distance_encoding:
            kernel_input_parts.append(rel)
        if 'dist' in self.distance_encoding:
            kernel_input_parts.append(dist)

        kernel_input = torch.stack(kernel_input_parts, dim=-1).view(B, N_q, N_o, -1)  # (B, N_q, N_o, pos_features)

        # --- Attention logits (multi-head) ---
        logits = self.kernel_mlp(kernel_input)  # (B, N_q, N_o, H)
        logits = logits.permute(0, 3, 1, 2)     # (B, H, N_q, N_o)

        # --- Distance masking ---
        if self.radius is not None:
            dist = torch.norm(rel, dim=-1)  # (B, N_q, N_o)
            logits = logits.masked_fill(
                dist.unsqueeze(1) > self.radius,
                float('-inf')
            )

        # --- Observation mask ---
        if obs_mask is not None:
            logits = logits.masked_fill(
                obs_mask.unsqueeze(1).unsqueeze(2) == 0,
                float('-inf')
            )

        weights = torch.softmax(logits, dim=-1)  # (B, H, N_q, N_o)

        # Layer Norm on h_obs before value projection
        h_obs = self.pre_ln(h_obs)

        # --- Values ---
        v = self.value_proj(h_obs)  # (B, N_o, H*d_head)
        v = v.view(B, N_o, self.heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)   # (B, H, N_o, d_head)

        # --- Aggregation ---
        h_query = torch.matmul(weights, v)  # (B, H, N_q, d_head)
        h_query = h_query.permute(0, 2, 1, 3).contiguous()
        h_query = h_query.view(B, N_q, self.latent_dim)  # (B, N_q, H*d_head)

        return h_query
        