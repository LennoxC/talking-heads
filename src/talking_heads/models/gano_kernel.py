import torch
import torch.nn as nn

class GANOKernel(nn.Module):
    def __init__(
        self,
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
        self.out_proj = nn.Linear(self.latent_dim, out_dim)

    def forward(
        self,
        h_obs,        # (N_o, d_h)
        pos_obs,      # (N_o, d_p)
        pos_query,    # (N_q, d_p)
        obs_mask=None, # (N_o,)
        obs_batch=None, # (N_o,)
        query_batch=None # (N_q,)
    ):
        device = pos_obs.device

        # --- Handle missing batch ---
        if obs_batch is None:
            obs_batch = torch.zeros(pos_obs.shape[0], dtype=torch.long, device=device)
        if query_batch is None:
            query_batch = torch.zeros(pos_query.shape[0], dtype=torch.long, device=device)

        N_q = pos_query.shape[0]
        N_o = pos_obs.shape[0]

        # --- Pairwise geometry (flattened) ---
        pos_q = pos_query.unsqueeze(1)   # (N_q, 1, d_p)
        pos_o = pos_obs.unsqueeze(0)     # (1, N_o, d_p)

        rel = pos_q - pos_o              # (N_q, N_o, d_p)
        dist = torch.norm(rel, dim=-1, keepdim=True)  # (N_q, N_o, 1)

        # expand
        pos_q_exp = pos_q.expand(-1, N_o, -1)
        pos_o_exp = pos_o.expand(N_q, -1, -1)

        # --- Kernel input ---
        kernel_input_parts = []
        if 'q_pos' in self.distance_encoding:
            kernel_input_parts.append(pos_q_exp)
        if 'o_pos' in self.distance_encoding:
            kernel_input_parts.append(pos_o_exp)
        if 'rel' in self.distance_encoding:
            kernel_input_parts.append(rel)
        if 'dist' in self.distance_encoding:
            kernel_input_parts.append(dist)

        kernel_input = torch.cat(kernel_input_parts, dim=-1)  # (N_q, N_o, pos_features)

        # --- Attention logits ---
        logits = self.kernel_mlp(kernel_input)  # (N_q, N_o, H)
        logits = logits.permute(2, 0, 1)        # (H, N_q, N_o)

        # --- Batch mask (CRITICAL) ---
        batch_mask = (query_batch.unsqueeze(1) == obs_batch.unsqueeze(0))  # (N_q, N_o)

        logits = logits.masked_fill(~batch_mask.unsqueeze(0), float('-inf'))

        # --- Radius mask ---
        if self.radius is not None:
            d = dist.squeeze(-1)  # (N_q, N_o)
            logits = logits.masked_fill(d.unsqueeze(0) > self.radius, float('-inf'))

        # --- Obs mask ---
        if obs_mask is not None:
            logits = logits.masked_fill(
                (~obs_mask).unsqueeze(0).unsqueeze(1),  # (1,1,N_o)
                float('-inf')
            )

        # --- Softmax ---
        weights = torch.softmax(logits, dim=-1)  # (H, N_q, N_o)

        # --- Values ---
        h_obs = self.pre_ln(h_obs)
        v = self.value_proj(h_obs)  # (N_o, H*d_head)
        v = v.view(N_o, self.heads, self.head_dim)
        v = v.permute(1, 0, 2)  # (H, N_o, d_head)

        # --- Aggregation ---
        h_query = torch.matmul(weights, v)  # (H, N_q, d_head)
        h_query = h_query.permute(1, 0, 2).contiguous()  # (N_q, H, d_head)
        h_query = h_query.view(N_q, self.latent_dim)     # (N_q, H*d_head)

        return h_query
        