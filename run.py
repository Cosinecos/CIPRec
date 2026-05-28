from __future__ import annotations

import argparse

from ciprec.config import load_config
from ciprec.trainer import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CIPRec")
    p.add_argument("--config", type=str, default="configs/ciprec_toy.yaml", help="YAML config path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)
