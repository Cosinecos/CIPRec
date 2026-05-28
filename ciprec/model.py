from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphGatedSessionEncoder(nn.Module):
    """A fast session encoder with local transition propagation + GRU + attention.

    This is a practical shared encoder for CIPRec. It keeps the interface close to
    GNN/session-graph encoders while avoiding expensive per-session sparse graph
    construction, so it is suitable for a clean open-source runnable version.
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
        layers: int = 1,
        use_layer_norm: bool = True,
        use_gru: bool = False,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.transition = nn.Linear(embed_dim * 3, embed_dim)
        self.transition_gate = nn.Linear(embed_dim * 3, embed_dim)
        self.use_gru = bool(use_gru)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=layers, batch_first=True) if self.use_gru else None
        self.token_proj = nn.Linear(embed_dim, hidden_dim)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.proj = nn.Linear(hidden_dim + embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_x = nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity()
        self.norm_out = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.item_embedding.weight)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0.0)

    @property
    def embedding(self) -> nn.Embedding:
        return self.item_embedding

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = seq.ne(0)
        x = self.item_embedding(seq)
        prev_x = torch.zeros_like(x)
        next_x = torch.zeros_like(x)
        prev_x[:, 1:] = x[:, :-1]
        next_x[:, :-1] = x[:, 1:]
        trans_in = torch.cat([x, prev_x, next_x], dim=-1)
        gate = torch.sigmoid(self.transition_gate(trans_in))
        msg = torch.tanh(self.transition(trans_in))
        x = self.norm_x(x + gate * msg)
        x = self.dropout(x)

        if self.use_gru:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=seq.size(1))
        else:
            out = self.token_proj(x)

        # Last item embedding as short-term intent anchor.
        last_pos = (lengths - 1).clamp_min(0).view(-1, 1, 1).expand(-1, 1, x.size(-1))
        last_emb = x.gather(dim=1, index=last_pos).squeeze(1)

        attn_in = torch.cat([out, last_emb.unsqueeze(1).expand(-1, out.size(1), -1)], dim=-1)
        attn_score = self.attn(attn_in).squeeze(-1)
        attn_score = attn_score.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(attn_score, dim=1)
        pooled = torch.sum(out * alpha.unsqueeze(-1), dim=1)
        u = self.proj(torch.cat([pooled, last_emb], dim=-1))
        return self.norm_out(u)


class CIPRec(nn.Module):
    """Context-Indexed Probabilistic Recommender.

    Implements:
      1) session encoding u_* = E(s)
      2) deterministic context summary c_* = Agg(phi(u_c, v_c))
      3) Gaussian prior p(eta|C_*) and posterior q(eta|C_*,u_*,v_*)
      4) multi-sample decoder q_* = mean_k psi(u_*, c_*, eta_k)
      5) full-ranking head score(i|s)=q_*^T e(i)
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 100,
        hidden_dim: int = 100,
        latent_dim: int = 100,
        dropout: float = 0.2,
        encoder_layers: int = 1,
        use_layer_norm: bool = True,
        use_gru: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim != embed_dim:
            # The paper notation uses one d-dimensional space. We support unequal
            # dims by projecting encoder output into embedding space.
            self.query_proj = nn.Linear(hidden_dim, embed_dim)
        else:
            self.query_proj = nn.Identity()
        self.num_items = int(num_items)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)

        self.encoder = GraphGatedSessionEncoder(
            num_items=num_items,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            layers=encoder_layers,
            use_layer_norm=use_layer_norm,
            use_gru=use_gru,
        )
        d = embed_dim
        self.phi = MLP(d * 2, d * 2, d, dropout=dropout)
        self.prior_net = MLP(d, d * 2, latent_dim * 2, dropout=dropout)
        self.posterior_net = MLP(d * 3, d * 2, latent_dim * 2, dropout=dropout)
        self.decoder = MLP(d * 2 + latent_dim, d * 2, d, dropout=dropout)
        self.query_norm = nn.LayerNorm(d) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    @property
    def item_embedding(self) -> nn.Embedding:
        return self.encoder.item_embedding

    def encode_session(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        u = self.encoder(seq, lengths)
        return self.query_proj(u)

    @staticmethod
    def _split_gaussian(params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = params.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, min=-8.0, max=6.0)
        return mu, logvar

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor, k: int) -> torch.Tensor:
        # [B, D] -> [B, K, D]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mu.size(0), int(k), mu.size(1), device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(1) + eps * std.unsqueeze(1)

    @staticmethod
    def gaussian_kl(q_mu: torch.Tensor, q_logvar: torch.Tensor, p_mu: torch.Tensor, p_logvar: torch.Tensor) -> torch.Tensor:
        # KL(N_q || N_p), averaged over batch.
        q_var = torch.exp(q_logvar)
        p_var = torch.exp(p_logvar)
        kl = 0.5 * (
            p_logvar - q_logvar + (q_var + (q_mu - p_mu).pow(2)) / p_var.clamp_min(1e-8) - 1.0
        )
        return kl.sum(dim=-1).mean()

    def summarize_context(self, u_c: torch.Tensor, v_c: torch.Tensor) -> torch.Tensor:
        # u_c/v_c: [B, C, d]
        pair = torch.cat([u_c, v_c], dim=-1)
        h = self.phi(pair)
        return h.mean(dim=1)

    def decode_query(self, u: torch.Tensor, c: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        # eta: [B, K, z]
        bsz, k, _ = eta.shape
        u_rep = u.unsqueeze(1).expand(-1, k, -1)
        c_rep = c.unsqueeze(1).expand(-1, k, -1)
        q_samples = self.decoder(torch.cat([u_rep, c_rep, eta], dim=-1))
        q = q_samples.mean(dim=1)
        # Residual keeps deterministic session signal and improves early training.
        return self.query_norm(q + u)

    def score_all_items(self, q: torch.Tensor) -> torch.Tensor:
        # Exclude padding row 0; scores correspond to item ids 1..num_items.
        emb = self.item_embedding.weight[1:]
        return q @ emb.T

    def forward(
        self,
        seq: torch.Tensor,
        lengths: torch.Tensor,
        context_u: torch.Tensor,
        context_v: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        n_samples: int = 10,
        beta: float = 1.0e-3,
    ) -> Dict[str, torch.Tensor]:
        u = self.encode_session(seq, lengths)
        c = self.summarize_context(context_u, context_v)
        p_mu, p_logvar = self._split_gaussian(self.prior_net(c))

        if self.training and target is not None:
            v_star = self.item_embedding(target)
            q_mu, q_logvar = self._split_gaussian(self.posterior_net(torch.cat([c, u, v_star], dim=-1)))
            eta = self._reparameterize(q_mu, q_logvar, k=n_samples)
            kl = self.gaussian_kl(q_mu, q_logvar, p_mu, p_logvar)
        else:
            eta = self._reparameterize(p_mu, p_logvar, k=n_samples)
            kl = torch.zeros((), device=seq.device)

        q_star = self.decode_query(u, c, eta)
        scores = self.score_all_items(q_star)
        out: Dict[str, torch.Tensor] = {
            "scores": scores,
            "q_star": q_star,
            "u_star": u,
            "c_star": c,
            "kl": kl,
            "prior_mu": p_mu,
            "prior_logvar": p_logvar,
        }
        if target is not None:
            labels = target - 1
            ce = F.cross_entropy(scores, labels)
            loss = ce + float(beta) * kl
            out.update({"loss": loss, "ce": ce})
        return out
