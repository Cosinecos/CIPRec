from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "data": {
        "root": "data/toy",
        "train_file": "train.jsonl",
        "valid_file": "valid.jsonl",
        "test_file": "test.jsonl",
        "max_len": 20,
        "num_workers": 0,
    },
    "model": {
        "embed_dim": 100,
        "hidden_dim": 100,
        "latent_dim": 100,
        "dropout": 0.2,
        "encoder_layers": 1,
        "use_layer_norm": True,
    },
    "context": {
        "top_c": 16,
        "pool_ratio": 0.10,
        "chunk_size": 4096,
        "refresh_each_epoch": True,
        "normalize": True,
    },
    "sampling": {
        "train_k": 5,
        "eval_k": 10,
    },
    "train": {
        "epochs": 20,
        "batch_size": 100,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-5,
        "beta": 1.0e-3,
        "grad_clip": 5.0,
        "patience": 5,
        "amp": False,
        "log_interval": 50,
    },
    "eval": {
        "topk": [5, 10, 20],
    },
    "output": {
        "dir": "outputs/toy",
        "save_best": True,
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str | Path | None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return cfg
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return _deep_update(cfg, user_cfg)


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
