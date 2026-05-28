# 🧠 CIPRec

> **Context-Indexed Probabilistic Neural Processes for Multimedia Session Retrieval**  
> 一个可运行、可复现实验、便于开源的 PyTorch 实现。

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Task-Session%20Recommendation-4f46e5" />
  <img src="https://img.shields.io/badge/Ranking-Full--Softmax-0f766e" />
  <img src="https://img.shields.io/badge/Status-Runnable-success" />
</p>

<p align="center">
  <b>Session Encoding → Context Indexing → Probabilistic Intent Inference → Multi-sample Decoding → Full-item Ranking</b>
</p>

---

## ✨ What is implemented?

CIPRec targets **session-based recommendation / multimedia session retrieval**.  
Given a short anonymous session \(s=\langle i_1,\dots,i_T\rangle\), the model builds a final ranking query \(q_*\) and retrieves the next item from the full item set.

This repository implements the full pipeline:

| Module | Code | Description |
|---|---|---|
| 🧩 Session Encoder | `ciprec/model.py::GraphGatedSessionEncoder` | Encodes session prefix into \(u_* = \mathcal{E}(s)\). |
| 🗂 Context Indexing | `ciprec/index.py` | Builds memory \(\mathcal{M}=\{(u,v)\}\) from training sessions and retrieves Top-C contexts. |
| 🌉 Deterministic Path | `CIPRec.summarize_context()` | Computes \(c_* = \mathrm{Agg}(\phi(u_c,v_c))\). |
| 🎲 Latent Path | `prior_net`, `posterior_net` | Learns Gaussian prior/posterior over latent intent \(\eta\). |
| 🔁 Multi-sample Decoder | `CIPRec.decode_query()` | Samples \(K\) latent intents and averages decoded queries. |
| 🎯 Full-ranking Head | `CIPRec.score_all_items()` | Scores all items by \(q_*^\top e(i)\). |
| 📊 Metrics | `ciprec/metrics.py` | HR@N, MRR@N, COV@N under full-ranking. |

---

## 🏗️ Pipeline

```mermaid
flowchart LR
  S[Session s=<i1,...,iT>] --> E[Session Encoder]
  E --> U[u_*]
  U --> CI[Context Indexing / Top-C]
  CI --> C[C_*={(u_c,v_c)}]
  C --> DS[Deterministic Summary c_*]
  DS --> P[Prior p(eta|C_*)]
  DS --> Q[Posterior q(eta|C_*,u_*,v_*) train only]
  P --> Z[eta^(k)]
  Q -. KL .-> L[ELBO]
  Z --> DEC[Decoder psi + Multi-sample]
  U --> DEC
  DS --> DEC
  DEC --> R[q_*]
  R --> HEAD[Full-item Ranking]
  HEAD --> TOP[Top-N / HR / MRR / COV]
  HEAD -. CE .-> L
```

---

## 📦 Project structure

```text
CIPRec/
├── ciprec/
│   ├── config.py          # YAML config loading and default config
│   ├── data.py            # JSONL dataset + dataloader
│   ├── index.py           # training-memory construction + Top-C retrieval
│   ├── metrics.py         # HR / MRR / COV full-ranking metrics
│   ├── model.py           # session encoder + CIPRec model
│   ├── trainer.py         # training / validation / test loop
│   └── utils.py           # seed, device, json, logging helpers
├── configs/
│   ├── ciprec_toy.yaml
│   ├── ciprec_retailrocket.yaml
│   ├── ciprec_diginetica.yaml
│   └── ciprec_nowplaying.yaml
├── tools/
│   ├── make_toy_data.py       # generate runnable synthetic data
│   ├── preprocess_sessions.py # convert raw click logs to JSONL
│   └── inspect_dataset.py
├── scripts/
│   ├── run_toy.sh
│   └── train_retailrocket.sh
├── data/
│   └── README.md
├── assets/
│   └── ciprec_pipeline.mmd
├── run.py
├── evaluate.py
├── inference.py
├── requirements.txt
└── README.md
```

---

## 🚀 Quick start

### 1. Create environment

```bash
conda create -n ciprec python=3.10 -y
conda activate ciprec
pip install -r requirements.txt
```

CPU 也能跑 toy demo；真实数据建议使用 CUDA。

### 2. Generate toy data

```bash
python tools/make_toy_data.py --out data/toy --train 800 --valid 200 --test 200 --num_items 200
```

### 3. Train CIPRec

```bash
python run.py --config configs/ciprec_toy.yaml
```

### 4. Evaluate checkpoint

```bash
python evaluate.py --checkpoint outputs/toy/best.pt --split test
```

### 5. Inference for custom sessions

```bash
python inference.py --checkpoint outputs/toy/best.pt --sessions "1,2,3;9,10,11" --topn 10
```

---

## 📚 Data format

The processed dataset should contain three files:

```text
data/your_dataset/
├── train.jsonl
├── valid.jsonl
└── test.jsonl
```

Each line is one prefix-to-next-item sample:

```json
{"session": [12, 91, 7, 33], "target": 104}
```

Important rules:

- Item ids must be positive integers.
- `0` is reserved for padding.
- Validation/test sessions are **not** used to build the context memory, avoiding leakage.

---

## 🧹 Convert raw click logs

If your raw file is a CSV with columns like `session_id,item_id,timestamp`, run:

```bash
python tools/preprocess_sessions.py \
  --csv /path/to/events.csv \
  --out data/custom \
  --session_col session_id \
  --item_col item_id \
  --time_col timestamp
```

Then create a config such as:

```yaml
data:
  root: data/custom
  max_len: 20

output:
  dir: outputs/custom
```

Run:

```bash
python run.py --config configs/your_config.yaml
```

---

## ⚙️ Key hyperparameters

| Name | Meaning | Recommended |
|---|---|---|
| `model.embed_dim` | item/query embedding dimension | 100 for real datasets |
| `model.latent_dim` | latent intent dimension | 100 |
| `context.top_c` | number of retrieved contexts | 16–64 |
| `context.pool_ratio` | ratio of training memory used for context retrieval | 0.1 by default |
| `sampling.train_k` | Monte Carlo samples during training | 3–5 |
| `sampling.eval_k` | Monte Carlo samples during evaluation | 10 |
| `train.beta` | KL regularization weight | 1e-3 |
| `train.amp` | mixed precision training | true on CUDA |

For large datasets, keep `pool_ratio=0.1` first. Increase `top_c` only after the training is stable.

---

## 🧪 Real dataset training template

Example for RetailRocket after preprocessing:

```bash
python run.py --config configs/ciprec_retailrocket.yaml
```

Expected outputs:

```text
outputs/retailrocket/
├── best.pt             # best checkpoint by valid MRR@10
├── config.yaml         # resolved config
├── history.json        # epoch-level training/validation log
└── test_metrics.json   # final test HR/MRR/COV
```

---

## 📊 Evaluation protocol

This implementation uses **full-ranking**, not sampled negatives:

```python
scores = q_star @ item_embedding.weight[1:].T
```

Then it computes:

- `HR@N`: whether the ground-truth target appears in Top-N.
- `MRR@N`: reciprocal rank of the ground-truth target within Top-N.
- `COV@N`: unique recommended items in Top-N divided by all items.

---

## 🧠 Implementation notes

1. **Context memory is built from train split only.**  
   This matches the no-leakage setting: validation/test sessions are never inserted into memory.

2. **The encoder is a strong runnable shared session encoder.**  
   It combines local transition propagation, GRU, and attention pooling. This keeps the code lightweight and stable while preserving the GNN/session-transition motivation.

3. **The probabilistic path uses posterior only during training.**  
   At inference time, only the context-conditioned prior is used.

4. **The ranking head is full-softmax.**  
   The loss is standard full-item cross entropy plus KL regularization.

---

## 🔧 Common issues

### CUDA out of memory

Reduce these values in the config:

```yaml
train:
  batch_size: 64
context:
  top_c: 16
sampling:
  train_k: 3
  eval_k: 5
```

### Training is slow on very large item sets

Full-softmax scores all items, so runtime scales with `#items`. First make sure CUDA is enabled, then reduce batch size. For paper-aligned full-ranking, do not replace it with sampled softmax unless you clearly report it as an approximation.

### Metrics look low on toy data

Toy data is intentionally noisy. Increase epochs:

```yaml
train:
  epochs: 10
```

---

## ✅ Minimal reproducibility checklist

```bash
python tools/make_toy_data.py --out data/toy
python tools/inspect_dataset.py --root data/toy
python run.py --config configs/ciprec_toy.yaml
python evaluate.py --checkpoint outputs/toy/best.pt --split test
```
