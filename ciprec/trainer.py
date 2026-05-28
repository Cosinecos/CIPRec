from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .config import save_config
from .data import build_datasets, make_loader
from .index import ContextBank, build_context_bank, retrieve_topc
from .metrics import MetricAccumulator
from .model import CIPRec
from .utils import AverageMeter, count_parameters, ensure_dir, get_device, set_seed, write_json


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(
    model: CIPRec,
    loader: torch.utils.data.DataLoader,
    bank: ContextBank,
    cfg: Dict,
    device: torch.device,
    split: str = "valid",
) -> Dict[str, float]:
    model.eval()
    topk = cfg["eval"]["topk"]
    meter = MetricAccumulator(topk=topk, num_items=model.num_items)
    losses = AverageMeter()
    bank = bank.to(device)
    iterator = tqdm(loader, desc=f"eval/{split}", leave=False)
    for batch in iterator:
        batch = _move_batch(batch, device)
        u = model.encode_session(batch["seq"], batch["lengths"])
        ctx = retrieve_topc(
            u,
            bank,
            top_c=cfg["context"]["top_c"],
            chunk_size=cfg["context"].get("chunk_size", 4096),
            normalize=cfg["context"].get("normalize", True),
        )
        out = model(
            batch["seq"],
            batch["lengths"],
            context_u=ctx["u_c"],
            context_v=ctx["v_c"],
            target=batch["target"],
            n_samples=cfg["sampling"].get("eval_k", 10),
            beta=cfg["train"].get("beta", 1e-3),
        )
        losses.update(out["loss"].item(), batch["target"].size(0))
        meter.update(out["scores"], batch["target"])
    result = meter.compute()
    result["loss"] = losses.avg
    return result


def train(cfg: Dict) -> Tuple[CIPRec, Dict[str, float]]:
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    out_dir = ensure_dir(cfg["output"]["dir"])
    save_config(cfg, out_dir / "config.yaml")

    train_set, valid_set, test_set, num_items = build_datasets(cfg["data"])
    batch_size = int(cfg["train"].get("batch_size", 100))
    num_workers = int(cfg["data"].get("num_workers", 0))
    train_loader = make_loader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    memory_loader = make_loader(train_set, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
    valid_loader = make_loader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = make_loader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = CIPRec(num_items=num_items, **cfg["model"]).to(device)
    print(f"[CIPRec] device={device} | items={num_items} | train={len(train_set)} | params={count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 1e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-5)),
    )
    scaler = GradScaler(enabled=bool(cfg["train"].get("amp", False)) and device.type == "cuda")

    best_metric = -math.inf
    best_epoch = 0
    bad_epochs = 0
    history = []
    bank = build_context_bank(
        model,
        memory_loader,
        device=device,
        pool_ratio=cfg["context"].get("pool_ratio", 0.10),
        normalize=cfg["context"].get("normalize", True),
    ).to(device)

    for epoch in range(1, int(cfg["train"].get("epochs", 20)) + 1):
        if epoch == 1 or cfg["context"].get("refresh_each_epoch", True):
            bank = build_context_bank(
                model,
                memory_loader,
                device=device,
                pool_ratio=cfg["context"].get("pool_ratio", 0.10),
                normalize=cfg["context"].get("normalize", True),
            ).to(device)
            print(f"[Epoch {epoch}] context memory size={bank.u.size(0)} | topC={cfg['context']['top_c']}")

        model.train()
        loss_meter, ce_meter, kl_meter = AverageMeter(), AverageMeter(), AverageMeter()
        iterator = tqdm(train_loader, desc=f"train/{epoch}", leave=False)
        for step, batch in enumerate(iterator, start=1):
            batch = _move_batch(batch, device)
            with torch.no_grad():
                u = model.encode_session(batch["seq"], batch["lengths"])
                ctx = retrieve_topc(
                    u,
                    bank,
                    top_c=cfg["context"]["top_c"],
                    chunk_size=cfg["context"].get("chunk_size", 4096),
                    normalize=cfg["context"].get("normalize", True),
                )
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                out = model(
                    batch["seq"],
                    batch["lengths"],
                    context_u=ctx["u_c"],
                    context_v=ctx["v_c"],
                    target=batch["target"],
                    n_samples=cfg["sampling"].get("train_k", 5),
                    beta=cfg["train"].get("beta", 1e-3),
                )
                loss = out["loss"]
            scaler.scale(loss).backward()
            if cfg["train"].get("grad_clip", 0) and float(cfg["train"].get("grad_clip", 0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()

            n = batch["target"].size(0)
            loss_meter.update(out["loss"].item(), n)
            ce_meter.update(out["ce"].item(), n)
            kl_meter.update(out["kl"].item(), n)
            iterator.set_postfix(loss=f"{loss_meter.avg:.4f}", ce=f"{ce_meter.avg:.4f}", kl=f"{kl_meter.avg:.3f}")

        valid_metrics = evaluate(model, valid_loader, bank, cfg, device, split="valid")
        monitor = valid_metrics.get("MRR@10", next(iter(valid_metrics.values())))
        row = {
            "epoch": epoch,
            "train_loss": loss_meter.avg,
            "train_ce": ce_meter.avg,
            "train_kl": kl_meter.avg,
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        }
        history.append(row)
        write_json({"history": history}, out_dir / "history.json")
        msg = " | ".join([f"{k}={v:.4f}" for k, v in valid_metrics.items() if k != "loss"])
        print(f"[Epoch {epoch}] train_loss={loss_meter.avg:.4f} | valid_loss={valid_metrics['loss']:.4f} | {msg}")

        if monitor > best_metric:
            best_metric = monitor
            best_epoch = epoch
            bad_epochs = 0
            if cfg["output"].get("save_best", True):
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg,
                        "num_items": num_items,
                        "best_epoch": best_epoch,
                        "best_metric": best_metric,
                    },
                    out_dir / "best.pt",
                )
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["train"].get("patience", 5)):
                print(f"[EarlyStop] best_epoch={best_epoch} best_valid_MRR@10={best_metric:.4f}")
                break

    ckpt = out_dir / "best.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
    bank = build_context_bank(
        model,
        memory_loader,
        device=device,
        pool_ratio=cfg["context"].get("pool_ratio", 0.10),
        normalize=cfg["context"].get("normalize", True),
    ).to(device)
    test_metrics = evaluate(model, test_loader, bank, cfg, device, split="test")
    write_json({"best_epoch": best_epoch, "test": test_metrics}, out_dir / "test_metrics.json")
    print("[Test] " + " | ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()]))
    return model, test_metrics
