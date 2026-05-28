#!/usr/bin/env bash
set -euo pipefail
# First convert your raw click-log CSV into data/retailrocket/*.jsonl, for example:
# python tools/preprocess_sessions.py --csv /path/to/events.csv --out data/retailrocket \
#   --session_col session_id --item_col item_id --time_col timestamp
python run.py --config configs/ciprec_retailrocket.yaml
