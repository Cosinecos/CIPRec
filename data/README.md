# Data format

CIPRec uses a simple JSONL format. Each line is one prefix-to-next-item sample:

```json
{"session": [12, 91, 7, 33], "target": 104}
```

Rules:

- Item ids must be positive integers.
- `0` is reserved for padding.
- `train.jsonl`, `valid.jsonl`, and `test.jsonl` must be placed under the dataset root.

You can generate a toy dataset:

```bash
python tools/make_toy_data.py --out data/toy
```

Or convert a click-log CSV:

```bash
python tools/preprocess_sessions.py \
  --csv /path/to/events.csv \
  --out data/custom \
  --session_col session_id \
  --item_col item_id \
  --time_col timestamp
```
