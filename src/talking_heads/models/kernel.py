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

        self.kernel_mlp = nn.Sequential(
            nn.Linear(self.position_encodings * pos_dim, latent_dim),
            getattr(nn, activation)(),
            nn.Linear(latent_dim, 1)
        )

        self.value_proj = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        h_obs,        # (N_o, d_in)
        x_obs,        # (N_o, d_in)
        pos_obs,      # (N_o, d_p)
        pos_query,    # (N_q, d_p)
        x_bg=None,    # (N_q, d_b)
        obs_mask=None # (N_o,)
    ):
        N_q = pos_query.shape[0] # query points
        N_o = pos_obs.shape[0]   # observation points

        pos_q = pos_query.unsqueeze(1)   # (N_q, 1, d_p)
        pos_o = pos_obs.unsqueeze(0)     # (1, N_o, d_p)

        rel = pos_q - pos_o              # (N_q, N_o, d_p)

        pos_q_exp = pos_q.expand(-1, N_o, -1)
        pos_o_exp = pos_o.expand(N_q, -1, -1)

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

        # ---- Aggregate ----
        v = self.value_proj(h_obs) # (N_o, d)
        h_query = weights @ v # (N_q, d)

        # ---- Background fusion ----
        if self.use_bg:
            h_bg = self.bg_encoder(x_bg)
            h_query = torch.cat([h_query, h_bg], dim=-1)

        return h_query
        
