from __future__ import annotations

import argparse
import json
from pathlib import Path


def scan(path: Path):
    n = 0
    max_item = 0
    lens = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            s = obj["session"]
            y = obj["target"]
            n += 1
            lens.append(len(s))
            max_item = max(max_item, y, max(s))
    return n, max_item, sum(lens) / max(1, len(lens)), max(lens)


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect processed JSONL dataset")
    p.add_argument("--root", default="data/toy")
    args = p.parse_args()
    root = Path(args.root)
    for name in ["train", "valid", "test"]:
        path = root / f"{name}.jsonl"
        n, max_item, avg_len, max_len = scan(path)
        print(f"{name:5s}: samples={n:,} | max_item={max_item:,} | avg_len={avg_len:.2f} | max_len={max_len}")


if __name__ == "__main__":
    main()
