from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_toy_generation(tmp_path: Path) -> None:
    out = tmp_path / "toy"
    subprocess.check_call([
        sys.executable,
        "tools/make_toy_data.py",
        "--out",
        str(out),
        "--train",
        "20",
        "--valid",
        "5",
        "--test",
        "5",
        "--num_items",
        "30",
    ])
    assert (out / "train.jsonl").exists()
