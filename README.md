# Text-to-SQL Fine-Tuning: Evaluating Whether Fine-Tuning Actually Helps

## Overview

This project fine-tunes a large language model (Qwen2.5-Coder-7B) to convert natural language questions into SQL queries — for example, turning *"How many singers are there?"* into `SELECT count(*) FROM singer`.

The focus of this project is rigorous evaluation, not just training a model and reporting a headline number. The core question addressed: **does fine-tuning measurably improve performance on this task, and if not, why?**

## Key Findings

- **Baseline (pre-fine-tuning) execution accuracy: 74.3%** on 1,034 held-out test questions
- **Post-fine-tuning execution accuracy: 71.7%** — no statistically significant improvement (overlapping 95% confidence intervals)
- **Error analysis** identified the dominant failure mode as schema-grounding errors — the model hallucinating plausible but incorrect column names (e.g., predicting `pet_type` when the actual schema column was `pettype`) — rather than syntax or logic errors
- **Exact-match rate improved substantially (16.2% → 44.5%)**, indicating the model learned to mimic the training set's SQL formatting conventions without a corresponding gain in query correctness
- A checkpoint comparison (testing an earlier training epoch) ruled out simple overfitting as the primary explanation
- **Inference footprint**: under 2GB peak GPU memory in 4-bit quantized inference, confirming practical deployability regardless of the accuracy outcome

## Why This Matters

Verifying whether a model change produces a genuine improvement — rather than assuming it does — is a core part of applied ML work. This project demonstrates that process end-to-end: a leakage-safe train/test split, a statistically grounded before/after comparison, root-cause error analysis, and a check against an alternative explanation (overfitting) before drawing conclusions.

## Methodology

1. **Data preparation** (`data_prep.py`) — Used the Spider dataset (natural language question → SQL query pairs across 160+ relational databases). Training and evaluation used disjoint sets of databases to prevent data leakage.
2. **Baseline evaluation** (`baseline_eval.py`) — Evaluated the unmodified base model on 1,034 held-out questions.
3. **Fine-tuning** (`finetune.py`) — Applied QLoRA (parameter-efficient fine-tuning) on 2,000 training examples.
4. **Post-training evaluation** (`evaluate_finetuned.py`) — Re-evaluated on the same held-out set for a direct comparison, including confidence intervals and regression analysis.
5. **Error analysis** (`error_analysis.py`) — Categorized failure modes to identify the root cause of the accuracy gap.
6. **Overfitting check** — Compared results against an earlier training checkpoint to rule out simple over-training as the explanation.
7. **Inference benchmarking** (`benchmark_direct.py`) — Measured GPU memory usage and latency to assess deployment feasibility.

## Repository Structure

```
├── src/
│   ├── data_prep.py             — Dataset preparation and leakage-safe splitting
│   ├── schema_utils.py          — Database schema serialization
│   ├── generate.py              — Model inference utilities
│   ├── baseline_eval.py         — Pre-fine-tuning evaluation
│   ├── finetune.py              — QLoRA fine-tuning
│   ├── evaluate_finetuned.py    — Post-fine-tuning evaluation and comparison
│   ├── eval_utils.py            — Execution-accuracy scoring against live databases
│   ├── error_analysis.py        — Failure mode categorization
│   └── benchmark_direct.py      — Inference latency and memory benchmarking
├── outputs/                     — Result artifacts (JSON)
└── requirements.txt
```

## Results Summary

| Model | Execution Accuracy | Exact Match | Notes |
|---|---|---|---|
| Base (pre-fine-tuning) | 74.3% [95% CI: 71.7–76.8%] | 16.2% | — |
| Fine-tuned | 71.7% [95% CI: 69.0–74.5%] | 44.5% | No statistically credible improvement |
| Earlier checkpoint | 72.0% (n=100 sample) | 38.0% | Rules out simple overfitting |

## Next Steps

- Implement schema retrieval to reduce column/table hallucination, the identified bottleneck
- Expand the training set for greater domain coverage
- Deployment via NVIDIA TensorRT-LLM was attempted for optimized inference serving; this was not completed due to environment constraints on free-tier cloud GPU infrastructure, documented transparently rather than omitted

## Reproduction

```bash
pip install -r requirements.txt

python src/data_prep.py --spider_dir /path/to/spider_data --n_train 2000
python src/baseline_eval.py --databases_dir /path/to/spider_data/database
python src/finetune.py --train_path data/train.jsonl
python src/evaluate_finetuned.py --databases_dir /path/to/spider_data/database
python src/error_analysis.py
python src/benchmark_direct.py --databases_dir /path/to/spider_data/database
```

