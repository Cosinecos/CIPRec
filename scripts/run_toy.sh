#!/usr/bin/env bash
set -euo pipefail
python tools/make_toy_data.py --out data/toy --train 800 --valid 200 --test 200 --num_items 200
python run.py --config configs/ciprec_toy.yaml
python evaluate.py --checkpoint outputs/toy/best.pt --split test
