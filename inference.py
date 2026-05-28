from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from ciprec.config import load_config
from ciprec.data import build_datasets, make_loader, collate_sessions
from ciprec.index import build_context_bank, retrieve_topc
from ciprec.model import CIPRec
from ciprec.utils import get_device


def parse_sessions(text: str) -> List[List[int]]:
    sessions = []
    for part in text.split(";"):
        items = [int(x.strip()) for x in part.split(",") if x.strip()]
        if items:
            sessions.append(items)
    if not sessions:
        raise ValueError("No valid sessions parsed. Example: --sessions '1,2,3;5,8,9'")
    return sessions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Top-N inference with CIPRec")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sessions", type=str, required=True, help="Example: '1,2,3;4,5,6'")
    p.add_argument("--topn", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = get_device("auto")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config") or load_config(Path(args.checkpoint).parent / "config.yaml")
    train_set, _, _, num_items = build_datasets(cfg["data"])
    model = CIPRec(num_items=num_items, **cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    memory_loader = make_loader(train_set, batch_size=int(cfg["train"].get("batch_size", 100)) * 2, shuffle=False)
    bank = build_context_bank(model, memory_loader, device, cfg["context"].get("pool_ratio", 0.1)).to(device)

    sessions = parse_sessions(args.sessions)
    pseudo = [(s, 1, i) for i, s in enumerate(sessions)]
    batch = collate_sessions(pseudo)
    seq = batch["seq"].to(device)
    lengths = batch["lengths"].to(device)
    with torch.no_grad():
        u = model.encode_session(seq, lengths)
        ctx = retrieve_topc(u, bank, top_c=cfg["context"].get("top_c", 16))
        out = model(seq, lengths, ctx["u_c"], ctx["v_c"], target=None, n_samples=cfg["sampling"].get("eval_k", 10))
        _, idx = torch.topk(out["scores"], k=args.topn, dim=1)
        pred = (idx + 1).cpu().tolist()
    for s, top_items in zip(sessions, pred):
        print(f"session={s} -> top{args.topn}={top_items}")
