from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class SessionDataset(Dataset):
    """JSONL dataset.

    Each line must be:
        {"session": [item_id_1, ..., item_id_T], "target": item_id}

    Item ids must be positive integers. 0 is reserved as padding.
    """

    def __init__(self, path: str | Path, max_len: int = 20) -> None:
        self.path = Path(path)
        self.max_len = int(max_len)
        self.samples: List[Tuple[List[int], int]] = []
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                session = [int(x) for x in obj["session"] if int(x) > 0]
                target = int(obj["target"])
                if len(session) == 0 or target <= 0:
                    continue
                if len(session) > self.max_len:
                    session = session[-self.max_len :]
                self.samples.append((session, target))
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], int, int]:
        session, target = self.samples[idx]
        return session, target, idx

    @property
    def max_item_id(self) -> int:
        m = 0
        for s, y in self.samples:
            m = max(m, y, max(s))
        return m


def collate_sessions(batch: Iterable[Tuple[List[int], int, int]]) -> Dict[str, torch.Tensor]:
    sessions, targets, indices = zip(*batch)
    lengths = torch.tensor([len(s) for s in sessions], dtype=torch.long)
    max_len = int(lengths.max().item())
    seq = torch.zeros(len(sessions), max_len, dtype=torch.long)
    for row, s in enumerate(sessions):
        seq[row, : len(s)] = torch.tensor(s, dtype=torch.long)
    return {
        "seq": seq,
        "lengths": lengths,
        "target": torch.tensor(targets, dtype=torch.long),
        "index": torch.tensor(indices, dtype=torch.long),
    }


def build_datasets(data_cfg: Dict) -> Tuple[SessionDataset, SessionDataset, SessionDataset, int]:
    root = Path(data_cfg["root"])
    max_len = int(data_cfg.get("max_len", 20))
    train = SessionDataset(root / data_cfg.get("train_file", "train.jsonl"), max_len=max_len)
    valid = SessionDataset(root / data_cfg.get("valid_file", "valid.jsonl"), max_len=max_len)
    test = SessionDataset(root / data_cfg.get("test_file", "test.jsonl"), max_len=max_len)
    num_items = max(train.max_item_id, valid.max_item_id, test.max_item_id)
    return train, valid, test, num_items


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_sessions,
        drop_last=False,
    )
