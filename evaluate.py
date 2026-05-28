from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ciprec.config import load_config
from ciprec.data import build_datasets, make_loader
from ciprec.index import build_context_bank
from ciprec.model import CIPRec
from ciprec.trainer import evaluate
from ciprec.utils import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved CIPRec checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = get_device("auto")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config") or load_config(Path(args.checkpoint).parent / "config.yaml")
    train_set, valid_set, test_set, num_items = build_datasets(cfg["data"])
    model = CIPRec(num_items=num_items, **cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    batch_size = int(cfg["train"].get("batch_size", 100))
    memory_loader = make_loader(train_set, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    eval_set = valid_set if args.split == "valid" else test_set
    eval_loader = make_loader(eval_set, batch_size=batch_size, shuffle=False, num_workers=0)
    bank = build_context_bank(model, memory_loader, device, cfg["context"].get("pool_ratio", 0.1)).to(device)
    metrics = evaluate(model, eval_loader, bank, cfg, device, split=args.split)
    print(metrics)
