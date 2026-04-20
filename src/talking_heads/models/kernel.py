import torch
import torch.nn as nn
from torch_geometric.nn import radius, knn
from torch_geometric.utils import softmax

class BipartiteKernel(nn.Module):
    def __init__(
        self,
        in_dim_obs,
        pos_dim,
        latent_dim,
        out_dim,
        bg_dim=None,
        radius=None,
        k=None,
        heads=4,
        head_dim=None,
        edge_mode: str = "multi_scale",  # "radius", "knn", "multi_scale"
        radii=None,  # list for multi-scale. E.g. for normalized coordinate space from -1 to 1, could be [0.1, 0.2, 0.4] (local, mid-range, global)
        activation='ReLU',
        distance_encoding=['rel', 'q_pos', 'o_pos', 'rbf'] # which distance features to use for attention. Can be any combination of: relative position (rel), query position (q_pos), observation position (o_pos), RBF of distance (rbf), Fourier features (fourier)
    ):
        super().__init__()

        self.use_bg = bg_dim is not None
        self.edge_mode = edge_mode
        self.radius = radius
        self.k = k
        self.radii = radii
        self.heads = heads

        self.head_dim = head_dim if head_dim is not None else latent_dim // heads
        self.total_dim = self.heads * self.head_dim

        self.distance_encoding = distance_encoding
        self.pos_dim = pos_dim

        # params for kernelized attention
        self.log_sigma = nn.Parameter(torch.tensor(-2.0))  # exp(-2) ≈ 0.135
        self.temperature = 1.5  # soften softmax

        # ---- Value projection ----
        self.value_proj = nn.Linear(latent_dim, self.total_dim)

        # ---- Attention MLP (per-edge) ----
        edge_input_dim = len(distance_encoding) * pos_dim

        self.kernel_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, latent_dim),
            getattr(nn, activation)(),
            nn.Linear(latent_dim, heads)  # one logit per head
        )

    # --------------------------------------------------
    # Edge construction
    # --------------------------------------------------
    def build_edges(self, pos_obs, pos_query, obs_batch, query_batch):
        if self.edge_mode == "radius":
            edge_index = radius(
                x=pos_obs,
                y=pos_query,
                r=self.radius,
                batch_x=obs_batch,
                batch_y=query_batch,
                max_num_neighbors=64  # safety cap
            )

        elif self.edge_mode == "knn":
            edge_index = knn(
                x=pos_obs,
                y=pos_query,
                k=self.k,
                batch_x=obs_batch,
                batch_y=query_batch
            )

        elif self.edge_mode == "multi_scale":
            if self.radii is None:
                raise ValueError("radii must be provided for multi_scale mode")
            if not isinstance(self.radii, (list, tuple)) or len(self.radii) == 0:
                raise ValueError("radii must be a non-empty list of floats")

            edge_list = []
            for r in self.radii:
                e = radius(
                    x=pos_obs,
                    y=pos_query,
                    r=r,
                    batch_x=obs_batch,
                    batch_y=query_batch,
                    max_num_neighbors=64
                )
                edge_list.append(e)
            edge_index = torch.cat(edge_list, dim=1)
        
        else:
            raise ValueError(f"Unknown edge_mode: {self.edge_mode}")

        return edge_index  # (2, E)

    # --------------------------------------------------
    # Distance encoding
    # --------------------------------------------------
    def compute_edge_features(self, pos_obs, pos_query, edge_index):
        dst, src = edge_index  # obs -> query

        pos_o = pos_obs[src]     # (E, d_p)
        pos_q = pos_query[dst]   # (E, d_p)

        rel = pos_q - pos_o

        feats = []
        if 'rel' in self.distance_encoding:
            feats.append(rel)
        if 'q_pos' in self.distance_encoding:
            feats.append(pos_q)
        if 'o_pos' in self.distance_encoding:
            feats.append(pos_o)
        if "rbf" in self.distance_encoding:
            # Example simple RBF encoding
            dist2 = (rel ** 2).sum(dim=-1, keepdim=True)
            feats.append(torch.exp(-dist2))

        if "fourier" in self.distance_encoding:
            # Example Fourier features
            freqs = self.fourier_freqs  # (F,)
            # (N_query, N_obs, p, F)
            angles = rel.unsqueeze(-1) * freqs
            feats.append(torch.sin(angles).flatten(-2))
            feats.append(torch.cos(angles).flatten(-2))

        edge_attr = torch.cat(feats, dim=-1)  # (E, d_edge)

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
        N_o = pos_obs.size(0)
        N_q = pos_query.size(0)

        # ---- Default batching ----
        if obs_batch is None:
            obs_batch = pos_obs.new_zeros(N_o, dtype=torch.long)
        if query_batch is None:
            query_batch = pos_query.new_zeros(N_q, dtype=torch.long)

        # ---- Build bipartite edges ----
        edge_index = self.build_edges(pos_obs, pos_query, obs_batch, query_batch)

        dst, src = edge_index  # obs -> query

        device = h_obs.device
        src = src.contiguous().to(device)
        dst = dst.contiguous().to(device)

        # ---- Mask invalid observations ----
        if obs_mask is not None:
            valid = obs_mask[src] > 0
            src = src[valid]
            dst = dst[valid]

        if src.numel() == 0:
            return torch.zeros(
                (N_q, self.heads * self.head_dim),
                device=h_obs.device,
                dtype=h_obs.dtype
            )

        # ---- Compute edge features ----
        edge_attr, rel = self.compute_edge_features(pos_obs, pos_query, (dst, src))

        # ---- Attention logits ----
        logits = self.kernel_mlp(edge_attr).contiguous()  # (E, heads)

        # ---- Distance kernel (smoothness) ----
        dist2 = (rel ** 2).sum(dim=-1, keepdim=True)  # (E, 1)

        sigma = torch.exp(self.log_sigma) + 1e-6
        kernel_log_weight = -dist2 / (2 * sigma**2)  # log Gaussian kernel

        # Broadcast to heads
        kernel_log_weight = kernel_log_weight.expand(-1, self.heads)

        logits = logits + kernel_log_weight

        if torch.isnan(logits).any():
            raise ValueError("NaNs in logits")

        # ---- Multi-head value projection ----
        v = self.value_proj(h_obs).view(N_o, self.heads, self.head_dim).contiguous()  # (N_o, heads * head_dim)
        v_src = v[src]  # (E, heads, head_dim)

        # ---- Normalize attention per query node ----
        # scatter over dst (queries)
        attn = softmax(logits, dst, num_nodes=N_q)  # (E, heads)

        attn = attn.unsqueeze(-1)  # (E, heads, 1)

        # ---- Weighted aggregation ----
        out = attn * v_src  # (E, heads, head_dim)

        h_query = out.new_zeros((N_q, self.heads, self.head_dim))
        h_query.index_add_(0, dst, out)

        # ---- Merge heads ----
        h_query = h_query.view(N_q, self.heads * self.head_dim)

        return h_query