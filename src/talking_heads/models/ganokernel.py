import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class GANOPlusKernel(nn.Module):
    def __init__(
        self,
        in_dim_obs,
        pos_dim,
        latent_dim,
        out_dim,
        heads=4,
        head_dim=None,
        radius=None,
        activation="ReLU",
        distance_encoding=("rel", "q_pos", "o_pos", "rbf"),
        q_chunk_size=4096,
        o_chunk_size=512,
    ):
        super().__init__()

        self.heads = heads
        self.head_dim = head_dim or (latent_dim // heads)
        self.total_dim = self.heads * self.head_dim

        self.radius = radius
        self.distance_encoding = distance_encoding

        self.q_chunk_size = q_chunk_size
        self.o_chunk_size = o_chunk_size

        # kernel MLP
        edge_dim = len(distance_encoding) * pos_dim
        self.kernel_mlp = nn.Sequential(
            nn.Linear(edge_dim, latent_dim),
            getattr(nn, activation)(),
            nn.Linear(latent_dim, heads)
        )

        # value projection
        self.value_proj = nn.Linear(latent_dim, self.total_dim)

        # distance scale
        self.log_sigma = nn.Parameter(torch.tensor(-2.0))

    # --------------------------------------------------
    # Pairwise feature builder (vectorized per block)
    # --------------------------------------------------
    def compute_features(self, pos_q, pos_o):
        # pos_q: (Q, d), pos_o: (O, d)

        q = pos_q.unsqueeze(1)  # (Q,1,d)
        o = pos_o.unsqueeze(0)  # (1,O,d)

        rel = q - o             # (Q,O,d)

        feats = []

        if "rel" in self.distance_encoding:
            feats.append(rel)

        if "q_pos" in self.distance_encoding:
            feats.append(q.expand(-1, o.size(1), -1))

        if "o_pos" in self.distance_encoding:
            feats.append(o.expand(q.size(0), -1, -1))

        if "rbf" in self.distance_encoding:
            dist2 = (rel ** 2).sum(dim=-1, keepdim=True)
            feats.append(torch.exp(-dist2))

        edge_attr = torch.cat(feats, dim=-1)
        return edge_attr, rel

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(
        self,
        h_obs,        # (N_o, d)
        pos_obs,      # (N_o, d_p)
        pos_query,    # (N_q, d_p)
        obs_mask=None,
        obs_batch=None,
        query_batch=None
    ):
        device = h_obs.device
        N_q = pos_query.size(0)
        N_o = pos_obs.size(0)

        # ---- value projection ----
        v = self.value_proj(h_obs).view(N_o, self.heads, self.head_dim)

        # ---- output buffer ----
        out = torch.zeros(
            (N_q, self.heads, self.head_dim),
            device=device,
            dtype=h_obs.dtype
        )

        # ---- chunk over queries ----
        for qi in range(0, N_q, self.q_chunk_size):
            qj = min(qi + self.q_chunk_size, N_q)

            pos_q = pos_query[qi:qj]
            Q = pos_q.size(0)

            # running numerator / denominator
            num = torch.zeros((Q, self.heads, self.head_dim), device=device)
            denom = torch.zeros((Q, self.heads), device=device)

            # ---- chunk over observations ----
            for oi in range(0, N_o, self.o_chunk_size):
                oj = min(oi + self.o_chunk_size, N_o)

                pos_o = pos_obs[oi:oj]
                v_o = v[oi:oj]

                # ---- features ----
                def chunk_fn(pos_q, pos_o, v_o):
                    edge_attr, rel = self.compute_features(pos_q, pos_o)

                    logits = self.kernel_mlp(edge_attr)  # (Q,O,H)

                    # ---- distance bias ----
                    dist2 = (rel ** 2).sum(dim=-1, keepdim=True)
                    sigma = torch.exp(self.log_sigma) + 1e-6
                    logits = logits - dist2 / (2 * sigma**2)

                    # ---- radius cutoff (optional) ----
                    if self.radius is not None:
                        dist = torch.sqrt(dist2 + 1e-12)
                        logits = logits.masked_fill(dist > self.radius, float("-inf"))

                    # ---- observation mask ----
                    if obs_mask is not None:
                        mask = obs_mask[oi:oj].view(1, -1, 1)
                        logits = logits.masked_fill(mask == 0, float("-inf"))

                    # ---- stable exp ----
                    max_logits = logits.max(dim=1, keepdim=True).values  # (Q,1,H)
                    exp_logits = torch.exp(logits - max_logits)

                    # ---- accumulate ----
                    denom = exp_logits.sum(dim=1)  # (Q,H)

                    num = torch.einsum(
                        "qoh,ohd->qhd",
                        exp_logits,
                        v_o
                    )

                    return num, denom
                
                num_chunk, denom_chunk = checkpoint(chunk_fn, pos_q, pos_o, v_o)
                num += num_chunk
                denom += denom_chunk

            # ---- normalize ----
            out[qi:qj] = num / (denom.unsqueeze(-1) + 1e-9)

        return out.view(N_q, self.heads * self.head_dim)