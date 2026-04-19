import torch
import torch.nn as nn

class GANOKernel(nn.Module):
    def __init__(
        self,
        in_dim_obs,
        pos_dim,
        latent_dim,
        out_dim,
        bg_dim=None,
        radius=None,
        activation='ReLU',
        distance_encoding=['q_pos', 'o_pos', 'rel']
    ):
        super().__init__()

        self.radius = radius
        self.use_bg = bg_dim is not None
        self.distance_encoding = distance_encoding
        self.position_encodings = len(distance_encoding)

        # TODO: multi-head attention
        self.kernel_mlp = nn.Sequential(
            nn.Linear(self.position_encodings * pos_dim, latent_dim),
            getattr(nn, activation)(),
            nn.Linear(latent_dim, 1)
        )

        self.value_proj = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        h_obs,        # (N_o, d_h) # TODO check shapes
        pos_obs,      # (N_o, d_p)
        pos_query,    # (N_q, d_p)
        h_bg=None,    # (N_q, d_h)
        obs_mask=None, # (N_o,)
        obs_batch=None, # (N_o,)
        query_batch=None # (N_q,)
    ):
        N_q = pos_query.shape[0] # query points
        N_o = pos_obs.shape[0]   # observation points

        # ---- Default batch handling (single graph fallback) ----
        if obs_batch is None:
            obs_batch = pos_obs.new_zeros(N_o, dtype=torch.long)
        if query_batch is None:
            query_batch = pos_query.new_zeros(N_q, dtype=torch.long)

        pos_q = pos_query.unsqueeze(1)   # (N_q, 1, d_p)
        pos_o = pos_obs.unsqueeze(0)     # (1, N_o, d_p)

        rel = pos_q - pos_o              # (N_q, N_o, d_p)

        pos_q_exp = pos_q.expand(-1, N_o, -1)
        pos_o_exp = pos_o.expand(N_q, -1, -1)

        # TODO: allow different types of distance encoding specified by self.distance_encoding.

        # kernel can see position and distance.
        #
        # consider adding:
        # dist = torch.norm(rel, dim=-1, keepdim=True)
        # or radial basis kernel might be better, use multiple scales and sigma_k:
        # rbf_k = exp(- (dist / sigma_k)^2).
        # sinusoidal distance encoding could also be used but this makes less sense. Could be useful for turbulence?
        # then:
        #
        # kernel_input = torch.cat([
        #     pos_q_exp,
        #     pos_o_exp,
        #     rel,
        #     dist or rbf_k
        # ], dim=-1)
        #
        # Gives the model access to radial distance between values
        #
        #
        # suggestion:
        # dist = torch.norm(rel, dim=-1, keepdim=True)
        # 
        # rbf = torch.exp(-dist**2 / sigma**2)   # multiple sigmas
        # 
        # fourier = torch.cat([
        #     torch.sin(freqs * rel),
        #     torch.cos(freqs * rel)
        # ], dim=-1)
        # 
        # kernel_input = torch.cat([
        #     rel,
        #     dist,
        #     rbf,
        #     fourier
        # ], dim=-1)
        
        kernel_input = torch.cat(
            [pos_q_exp, pos_o_exp, rel], dim=-1
        )

        # compute attention weights
        logits = self.kernel_mlp(kernel_input).squeeze(-1) # (N_q, N_o). Row is query positions, cols are obs nodes
        same_batch = (query_batch.unsqueeze(1) == obs_batch.unsqueeze(0))
        logits = logits.masked_fill(~same_batch, float('-inf'))

        # ---- Distance masking (LOCAL ATTENTION) ----
        if self.radius is not None:
            dist = torch.norm(rel, dim=-1)
            logits = logits.masked_fill(dist > self.radius, float('-inf'))

        # ---- Observation mask ----
        if obs_mask is not None:
            logits = logits.masked_fill(
                obs_mask.unsqueeze(0) == 0,
                float('-inf')
            )

        weights = torch.softmax(logits, dim=1) # (N_q, N_o)
        weights = torch.nan_to_num(weights, nan=0.0) # if all logits are -inf, softmax returns nan, so convert these to 0 weights. This could happen if a query has no valid neighbors (e.g. due to masking or distance thresholding).

        # ---- Aggregate ----
        v = self.value_proj(h_obs) # (N_o, d)
        h_query = weights @ v # (N_q, d)

        # ---- Background fusion ----
        if self.use_bg:
            h_query = torch.cat([h_query, h_bg], dim=-1)

        return h_query
        
import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_sum
from torch_geometric.nn import radius, knn
#from torch_geometric.nn.pool import radius, knn


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
        edge_mode: str = "radius",  # "radius", "knn", "multi_scale"
        radii=None,  # list for multi-scale. E.g. for normalized coordinate space from -1 to 1, could be [0.1, 0.2, 0.4] (local, mid-range, global)
        activation='ReLU',
        distance_encoding=['rel'],
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
        src, dst = edge_index  # obs -> query

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
        h_bg=None,    # (N_q, d)
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
        src, dst = edge_index  # obs -> query

        # ---- Mask invalid observations ----
        if obs_mask is not None:
            valid = obs_mask[src] > 0
            src = src[valid]
            dst = dst[valid]

        # ---- Compute edge features ----
        edge_attr, rel = self.compute_edge_features(pos_obs, pos_query, (src, dst))

        # ---- Attention logits ----
        logits = self.kernel_mlp(edge_attr)  # (E, heads)

        # ---- Multi-head value projection ----
        v = self.value_proj(h_obs)  # (N_o, heads * head_dim)
        v = v.view(N_o, self.heads, self.head_dim)

        v_src = v[src]  # (E, heads, head_dim)

        # ---- Normalize attention per query node ----
        # scatter over dst (queries)
        attn = scatter_softmax(logits, dst, dim=0)  # (E, heads)

        attn = attn.unsqueeze(-1)  # (E, heads, 1)

        # ---- Weighted aggregation ----
        out = attn * v_src  # (E, heads, head_dim)

        h_query = scatter_sum(out, dst, dim=0, dim_size=N_q)  # (N_q, heads, head_dim)

        # ---- Merge heads ----
        h_query = h_query.view(N_q, self.heads * self.head_dim)

        # ---- Background fusion ----
        if self.use_bg:
            h_query = torch.cat([h_query, h_bg], dim=-1)

        return h_query