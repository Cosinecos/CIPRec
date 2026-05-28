from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class ContextBank:
    u: torch.Tensor  # [M, d]
    v: torch.Tensor  # [M, d]
    target: torch.Tensor  # [M]

    def to(self, device: torch.device) -> "ContextBank":
        return ContextBank(self.u.to(device), self.v.to(device), self.target.to(device))


@torch.no_grad()
def build_context_bank(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pool_ratio: float = 0.10,
    normalize: bool = True,
    quiet: bool = False,
) -> ContextBank:
    """Encode training sessions into memory M={(u, v)}.

    v is the embedding of the next clicked item. The caller should pass a training
    loader only; validation/test sessions must not be used to prevent leakage.
    """
    model.eval()
    all_u, all_v, all_y = [], [], []
    iterator = loader if quiet else tqdm(loader, desc="build context memory", leave=False)
    for batch in iterator:
        seq = batch["seq"].to(device)
        lengths = batch["lengths"].to(device)
        y = batch["target"].to(device)
        u = model.encode_session(seq, lengths)
        v = model.item_embedding(y)
        if normalize:
            u = F.normalize(u, dim=-1)
            v = F.normalize(v, dim=-1)
        all_u.append(u.detach().cpu())
        all_v.append(v.detach().cpu())
        all_y.append(y.detach().cpu())
    u = torch.cat(all_u, dim=0)
    v = torch.cat(all_v, dim=0)
    target = torch.cat(all_y, dim=0)

    ratio = float(pool_ratio)
    if 0 < ratio < 1.0:
        m = max(1, int(math.ceil(u.size(0) * ratio)))
        perm = torch.randperm(u.size(0))[:m]
        u, v, target = u[perm], v[perm], target[perm]
    return ContextBank(u=u, v=v, target=target)


@torch.no_grad()
def retrieve_topc(
    query_u: torch.Tensor,
    bank: ContextBank,
    top_c: int,
    chunk_size: int = 4096,
    normalize: bool = True,
) -> Dict[str, torch.Tensor]:
    """Retrieve Top-C contexts for each query by cosine or dot-product similarity."""
    if normalize:
        q = F.normalize(query_u, dim=-1)
        mem = F.normalize(bank.u, dim=-1)
    else:
        q = query_u
        mem = bank.u
    top_c = min(int(top_c), mem.size(0))
    best_scores = []
    best_indices = []
    for start in range(0, mem.size(0), int(chunk_size)):
        sim = q @ mem[start : start + int(chunk_size)].T
        vals, idx = torch.topk(sim, k=min(top_c, sim.size(1)), dim=1)
        best_scores.append(vals)
        best_indices.append(idx + start)
    scores = torch.cat(best_scores, dim=1)
    indices = torch.cat(best_indices, dim=1)
    vals, local = torch.topk(scores, k=top_c, dim=1)
    idx = torch.gather(indices, dim=1, index=local)
    return {
        "u_c": bank.u[idx],
        "v_c": bank.v[idx],
        "indices": idx,
        "scores": vals,
    }
