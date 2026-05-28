from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple


def make_session(num_items: int, num_topics: int, min_len: int, max_len: int) -> Tuple[List[int], int]:
    topic_size = num_items // num_topics
    topic = random.randrange(num_topics)
    start = topic * topic_size + 1
    end = start + topic_size - 1
    length = random.randint(min_len, max_len)
    # Topic-dominant sequence with mild noise and a deterministic-ish target.
    session = []
    for t in range(length):
        if random.random() < 0.15:
            session.append(random.randint(1, num_items))
        else:
            session.append(random.randint(start, end))
    # Target is topic-consistent but depends on recent item, so model can learn.
    last = session[-1]
    if start <= last <= end:
        offset = ((last - start + random.choice([1, 2, 3])) % topic_size)
        target = start + offset
    else:
        target = random.randint(start, end)
    return session, target


def write_jsonl(samples: List[Tuple[List[int], int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for session, target in samples:
            f.write(json.dumps({"session": session, "target": target}, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a small synthetic session dataset")
    p.add_argument("--out", type=str, default="data/toy")
    p.add_argument("--num_items", type=int, default=200)
    p.add_argument("--num_topics", type=int, default=20)
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--valid", type=int, default=400)
    p.add_argument("--test", type=int, default=400)
    p.add_argument("--min_len", type=int, default=3)
    p.add_argument("--max_len", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)
    out = Path(args.out)
    all_samples = [
        make_session(args.num_items, args.num_topics, args.min_len, args.max_len)
        for _ in range(args.train + args.valid + args.test)
    ]
    write_jsonl(all_samples[: args.train], out / "train.jsonl")
    write_jsonl(all_samples[args.train : args.train + args.valid], out / "valid.jsonl")
    write_jsonl(all_samples[args.train + args.valid :], out / "test.jsonl")
    meta = vars(args)
    with (out / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Toy data written to {out.resolve()}")


if __name__ == "__main__":
    main()
