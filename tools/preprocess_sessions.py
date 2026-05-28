from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split


def write_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert click log CSV to CIPRec JSONL")
    p.add_argument("--csv", required=True, help="CSV with session_id,item_id,timestamp columns")
    p.add_argument("--out", default="data/custom")
    p.add_argument("--session_col", default="session_id")
    p.add_argument("--item_col", default="item_id")
    p.add_argument("--time_col", default="timestamp")
    p.add_argument("--min_len", type=int, default=2)
    p.add_argument("--valid_ratio", type=float, default=0.10)
    p.add_argument("--test_ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    need = [args.session_col, args.item_col, args.time_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Existing columns: {list(df.columns)}")

    # Re-map item ids to 1..N, reserve 0 for padding.
    item_values = sorted(df[args.item_col].dropna().unique().tolist())
    item2id = {raw: i + 1 for i, raw in enumerate(item_values)}
    df["_item_id"] = df[args.item_col].map(item2id).astype(int)
    df = df.sort_values([args.session_col, args.time_col])

    samples = []
    for sid, g in df.groupby(args.session_col):
        items = g["_item_id"].tolist()
        if len(items) < args.min_len:
            continue
        # Prefix expansion: [i1]->i2, [i1,i2]->i3, ...
        for t in range(1, len(items)):
            samples.append({"session": items[:t], "target": items[t]})

    train_rows, tmp_rows = train_test_split(
        samples,
        test_size=args.valid_ratio + args.test_ratio,
        random_state=args.seed,
        shuffle=True,
    )
    rel_test = args.test_ratio / (args.valid_ratio + args.test_ratio)
    valid_rows, test_rows = train_test_split(tmp_rows, test_size=rel_test, random_state=args.seed, shuffle=True)

    out = Path(args.out)
    write_jsonl(train_rows, out / "train.jsonl")
    write_jsonl(valid_rows, out / "valid.jsonl")
    write_jsonl(test_rows, out / "test.jsonl")
    with (out / "item_map.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in item2id.items()}, f, ensure_ascii=False, indent=2)
    print(f"Processed {len(samples)} prefix samples, {len(item2id)} items -> {out.resolve()}")


if __name__ == "__main__":
    main()
