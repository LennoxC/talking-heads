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
        chunking_factor=1,
        edge_mode: str = "multi_scale",
        radii=None,
        activation='ReLU',
        distance_encoding=['rel', 'q_pos', 'o_pos', 'rbf']
    ):
        super().__init__()

        self.use_bg = bg_dim is not None
        self.edge_mode = edge_mode
        self.radius = radius
        self.k = k
        self.radii = radii
        self.heads = heads
        self.chunking_factor = max(1, int(chunking_factor))

        self.head_dim = head_dim if head_dim is not None else latent_dim // heads
        self.total_dim = self.heads * self.head_dim

        self.distance_encoding = distance_encoding
        self.pos_dim = pos_dim

        # kernel params
        self.log_sigma = nn.Parameter(torch.tensor(-2.0))
        self.temperature = 1.5

        # projections
        self.value_proj = nn.Linear(latent_dim, self.total_dim)

        edge_input_dim = len(distance_encoding) * pos_dim

        self.kernel_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, latent_dim),
            getattr(nn, activation)(),
            nn.Linear(latent_dim, heads)
        )

    # --------------------------------------------------
    # Sparse edge construction (unchanged)
    # --------------------------------------------------
    def build_edges(self, pos_obs, pos_query, obs_batch, query_batch):
        if self.edge_mode == "radius":
            return radius(
                x=pos_obs,
                y=pos_query,
                r=self.radius,
                batch_x=obs_batch,
                batch_y=query_batch,
                max_num_neighbors=64
            )

        elif self.edge_mode == "knn":
            return knn(
                x=pos_obs,
                y=pos_query,
                k=self.k,
                batch_x=obs_batch,
                batch_y=query_batch
            )

        elif self.edge_mode == "multi_scale":
            if self.radii is None:
                raise ValueError("radii must be provided")

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

            return torch.cat(edge_list, dim=1)

        else:
            raise ValueError(f"Edge mode {self.edge_mode} not supported here")

    # --------------------------------------------------
    # Dense (matrix-based) kernel
    # --------------------------------------------------
    def forward_full(self, h_obs, pos_obs, pos_query):
        """
        Exact dense attention computed in query chunks.
        This preserves the full-mode result while reducing peak memory.
        """
        N_o = pos_obs.size(0)
        N_q = pos_query.size(0)

        def compute_chunk(chunk_pos_query):
            chunk_size = chunk_pos_query.size(0)

            pos_o = pos_obs.unsqueeze(0)  # (1, N_o, d_p)
            pos_q = chunk_pos_query.unsqueeze(1)  # (chunk, 1, d_p)
            rel = pos_q - pos_o  # (chunk, N_o, d_p)

            feats = []
            if 'rel' in self.distance_encoding:
                feats.append(rel)
            if 'q_pos' in self.distance_encoding:
                feats.append(pos_q.expand(-1, N_o, -1))
            if 'o_pos' in self.distance_encoding:
                feats.append(pos_o.expand(chunk_size, -1, -1))
            if "rbf" in self.distance_encoding:
                dist2 = (rel ** 2).sum(dim=-1, keepdim=True)
                feats.append(torch.exp(-dist2))

            edge_attr = torch.cat(feats, dim=-1)
            logits = self.kernel_mlp(edge_attr)

            dist2 = (rel ** 2).sum(dim=-1, keepdim=True)
            sigma = torch.exp(self.log_sigma) + 1e-6
            kernel_log_weight = -dist2 / (2 * sigma**2)
            logits = (logits + kernel_log_weight.expand(-1, -1, self.heads)) / self.temperature

            attn = torch.softmax(logits, dim=1)
            v = self.value_proj(h_obs).view(N_o, self.heads, self.head_dim)

            return torch.einsum("qoh,ohd->qhd", attn, v)

        if N_q == 0:
            return torch.zeros((0, self.heads * self.head_dim), device=h_obs.device, dtype=h_obs.dtype)

        chunk_size = max(1, (N_q + self.chunking_factor - 1) // self.chunking_factor)
        out = torch.zeros((N_q, self.heads, self.head_dim), device=h_obs.device, dtype=h_obs.dtype)

        for start in range(0, N_q, chunk_size):
            end = min(start + chunk_size, N_q)
            out[start:end] = compute_chunk(pos_query[start:end])

        return out.reshape(N_q, self.heads * self.head_dim)

    # --------------------------------------------------
    # Sparse forward (unchanged logic)
    # --------------------------------------------------
    def forward_sparse(
        self,
        h_obs,
        pos_obs,
        pos_query,
        obs_batch,
        query_batch,
        obs_mask
    ):
        N_o = pos_obs.size(0)
        N_q = pos_query.size(0)

        edge_index = self.build_edges(
            pos_obs, pos_query, obs_batch, query_batch
        )

        dst, src = edge_index

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

        pos_o = pos_obs[src]
        pos_q = pos_query[dst]
        rel = pos_q - pos_o

        feats = []
        if 'rel' in self.distance_encoding:
            feats.append(rel)
        if 'q_pos' in self.distance_encoding:
            feats.append(pos_q)
        if 'o_pos' in self.distance_encoding:
            feats.append(pos_o)
        if "rbf" in self.distance_encoding:
            dist2 = (rel ** 2).sum(dim=-1, keepdim=True)
            feats.append(torch.exp(-dist2))

        edge_attr = torch.cat(feats, dim=-1)

        logits = self.kernel_mlp(edge_attr)

        dist2 = (rel ** 2).sum(dim=-1, keepdim=True)
        sigma = torch.exp(self.log_sigma) + 1e-6
        kernel_log_weight = -dist2 / (2 * sigma**2)

        logits = logits + kernel_log_weight.expand(-1, self.heads)

        v = self.value_proj(h_obs).view(N_o, self.heads, self.head_dim)
        v_src = v[src]

        attn = softmax(logits, dst, num_nodes=N_q).unsqueeze(-1)

        out = attn * v_src

        h_query = torch.zeros(
            (N_q, self.heads, self.head_dim),
            device=h_obs.device,
            dtype=h_obs.dtype
        )

        h_query.index_add_(0, dst, out)

        return h_query.view(N_q, self.heads * self.head_dim)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(
        self,
        h_obs,
        pos_obs,
        pos_query,
        obs_mask=None,
        obs_batch=None,
        query_batch=None
    ):
        N_o = pos_obs.size(0)
        N_q = pos_query.size(0)

        if obs_batch is None:
            obs_batch = pos_obs.new_zeros(N_o, dtype=torch.long)
        if query_batch is None:
            query_batch = pos_query.new_zeros(N_q, dtype=torch.long)

        if self.edge_mode == "full":
            return self.forward_full(h_obs, pos_obs, pos_query)
        else:
            return self.forward_sparse(
                h_obs,
                pos_obs,
                pos_query,
                obs_batch,
                query_batch,
                obs_mask
            )