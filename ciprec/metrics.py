from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import torch


@torch.no_grad()
def topk_metrics_from_scores(
    scores: torch.Tensor,
    targets: torch.Tensor,
    topk: Iterable[int],
    num_items: int,
) -> Dict[str, float]:
    """Compute HR/MRR/COV for one batch.

    scores is shaped [B, num_items] and corresponds to item ids 1..num_items.
    targets are original positive ids in [1, num_items].
    """
    ks: List[int] = sorted(int(k) for k in topk)
    max_k = max(ks)
    _, top_idx = torch.topk(scores, k=max_k, dim=1)
    top_items = top_idx + 1
    target = targets.view(-1, 1)
    out: Dict[str, float] = {}
    for k in ks:
        pred_k = top_items[:, :k]
        hit_mat = pred_k.eq(target)
        hit = hit_mat.any(dim=1).float()
        out[f"HR@{k}"] = hit.mean().item()

        # First rank within top-k; zero if not found.
        ranks = torch.arange(1, k + 1, device=scores.device).view(1, -1).float()
        rr = (hit_mat[:, :k].float() / ranks).sum(dim=1)
        out[f"MRR@{k}"] = rr.mean().item()

        cov = torch.unique(pred_k).numel() / float(num_items)
        out[f"COV@{k}"] = cov
    return out


class MetricAccumulator:
    def __init__(self, topk: Iterable[int], num_items: int) -> None:
        self.topk = sorted(int(k) for k in topk)
        self.num_items = int(num_items)
        self.sum = defaultdict(float)
        self.count = 0
        self.coverage_sets = {k: set() for k in self.topk}

    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> None:
        bsz = targets.size(0)
        ks = self.topk
        max_k = max(ks)
        _, top_idx = torch.topk(scores, k=max_k, dim=1)
        top_items = (top_idx + 1).detach().cpu()
        target = targets.detach().cpu().view(-1, 1)
        for k in ks:
            pred_k = top_items[:, :k]
            hit_mat = pred_k.eq(target)
            hit = hit_mat.any(dim=1).float()
            ranks = torch.arange(1, k + 1).view(1, -1).float()
            rr = (hit_mat[:, :k].float() / ranks).sum(dim=1)
            self.sum[f"HR@{k}"] += hit.sum().item()
            self.sum[f"MRR@{k}"] += rr.sum().item()
            self.coverage_sets[k].update(pred_k.reshape(-1).tolist())
        self.count += bsz

    def compute(self) -> Dict[str, float]:
        out = {}
        for k in self.topk:
            out[f"HR@{k}"] = self.sum[f"HR@{k}"] / max(1, self.count)
            out[f"MRR@{k}"] = self.sum[f"MRR@{k}"] / max(1, self.count)
            out[f"COV@{k}"] = len(self.coverage_sets[k]) / float(self.num_items)
        return out
