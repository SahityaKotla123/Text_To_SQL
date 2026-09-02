"""
Day 1-2: Data preparation.

- Loads Spider's OFFICIAL train/dev split from LOCAL JSON files (the real
  Spider release: train_spider.json, train_others.json, dev.json, tables.json)
  — NOT the Hugging Face mirror, which is missing schema fields entirely.
- Stratified-samples 1,000-3,000 examples from the training split, balanced
  across databases and (approximate) difficulty.
- Leaves the dev split completely untouched — it already contains databases
  unseen in training, which is what gives you a leakage-free eval set.
- Builds (prompt, gold_sql, db_id, difficulty) records using BOTH minimal and
  rich schema serialization, so Day 5's ablation can reuse this file directly.

Usage:
    python src/data_prep.py --spider_dir /content/drive/MyDrive/text2sql_phase1/spider_data \
        --n_train 2000 --seed 42
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from schema_utils import SchemaSpec, serialize_minimal, serialize_rich, build_prompt
from difficulty import classify_hardness


def load_spider_official(spider_dir: Path):
    """
    Loads Spider's official split from local JSON files that ship with the
    real Spider release (the one that also contains the .sqlite databases).
    We do NOT re-split by database ourselves — Spider's official split
    already separates databases correctly, and re-splitting on our own would
    risk reintroducing the leakage the review feedback warned about.

    train_spider.json + train_others.json together make up Spider's full
    official training pool (this matches how the Spider papers/leaderboard
    define "train"). dev.json is the official dev split — untouched.
    """
    with open(spider_dir / "train_spider.json") as f:
        train = json.load(f)
    train_others_path = spider_dir / "train_others.json"
    if train_others_path.exists():
        with open(train_others_path) as f:
            train = train + json.load(f)
    with open(spider_dir / "dev.json") as f:
        dev = json.load(f)
    return train, dev


def load_schemas(spider_dir: Path):
    """
    Builds a dict[db_id] -> SchemaSpec directly from tables.json, which maps
    cleanly onto SchemaSpec's fields:
        table_names_original  -> table_names
        column_names_original -> column_names  (already [table_idx, col_name] pairs)
        column_types          -> column_types
        primary_keys          -> primary_keys  (global column indices)
        foreign_keys          -> foreign_keys  ([col_a, col_b] index pairs)
    """
    with open(spider_dir / "tables.json") as f:
        tables = json.load(f)

    schemas = {}
    for entry in tables:
        db_id = entry["db_id"]
        schemas[db_id] = SchemaSpec(
            db_id=db_id,
            table_names=entry["table_names_original"],
            column_names=[tuple(c) for c in entry["column_names_original"]],
            column_types=entry["column_types"],
            primary_keys=entry["primary_keys"],
            foreign_keys=[tuple(fk) for fk in entry["foreign_keys"]],
        )
    return schemas


def stratified_sample(train, n_target: int, seed: int = 42):
    """
    Stratified sample across (db_id, difficulty) buckets so no single
    database or difficulty tier dominates the training set.
    """
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for i, row in enumerate(train):
        diff = classify_hardness(row["query"])
        buckets[(row["db_id"], diff)].append(i)

    for key in buckets:
        rng.shuffle(buckets[key])

    # Round-robin draw across buckets until we hit n_target
    keys = list(buckets.keys())
    rng.shuffle(keys)
    selected = []
    pointer = {k: 0 for k in keys}
    while len(selected) < n_target:
        progressed = False
        for k in keys:
            if pointer[k] < len(buckets[k]):
                selected.append(buckets[k][pointer[k]])
                pointer[k] += 1
                progressed = True
                if len(selected) >= n_target:
                    break
        if not progressed:
            break  # exhausted all buckets before hitting n_target
    return selected


def build_records(dataset, indices, schemas, split_name: str):
    records = []
    skipped = 0
    for i in indices:
        row = dataset[i]
        db_id = row["db_id"]
        schema = schemas.get(db_id)
        if schema is None:
            skipped += 1
            continue
        minimal = serialize_minimal(schema)
        rich = serialize_rich(schema)
        records.append({
            "split": split_name,
            "db_id": db_id,
            "question": row["question"],
            "gold_sql": row["query"],
            "difficulty": classify_hardness(row["query"]),
            "prompt_minimal": build_prompt(row["question"], minimal),
            "prompt_rich": build_prompt(row["question"], rich),
        })
    if skipped:
        print(f"[{split_name}] skipped {skipped} rows with missing schema info")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider_dir", type=str, required=True,
                         help="Path to the folder containing train_spider.json, dev.json, "
                              "tables.json, etc. (e.g. .../text2sql_phase1/spider_data)")
    parser.add_argument("--n_train", type=int, default=2000,
                         help="Number of stratified examples to sample from Spider's official train split (1000-3000 recommended)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    spider_dir = Path(args.spider_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Spider's official train/dev split from {spider_dir}...")
    train, dev = load_spider_official(spider_dir)
    print(f"  official train size: {len(train)}, official dev size: {len(dev)}")

    print("Building schema index from tables.json...")
    schemas = load_schemas(spider_dir)
    print(f"  {len(schemas)} unique databases indexed")

    print(f"Stratified sampling {args.n_train} examples from official train split...")
    train_idx = stratified_sample(train, args.n_train, seed=args.seed)
    print(f"  sampled {len(train_idx)} examples")

    train_records = build_records(train, train_idx, schemas, "train")
    # Dev split used WHOLE and UNCHANGED — this is Spider's official unseen-DB eval set.
    dev_records = build_records(dev, range(len(dev)), schemas, "dev")

    with open(out_dir / "train.jsonl", "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "dev.jsonl", "w") as f:
        for r in dev_records:
            f.write(json.dumps(r) + "\n")

    # Quick sanity report: difficulty distribution, db overlap check
    train_dbs = {r["db_id"] for r in train_records}
    dev_dbs = {r["db_id"] for r in dev_records}
    overlap = train_dbs & dev_dbs
    print(f"\nTrain DBs: {len(train_dbs)} | Dev DBs: {len(dev_dbs)} | Overlap: {len(overlap)}")
    print("(Spider's official split is DB-disjoint by construction — overlap should be 0.")
    print(" If it isn't, something went wrong upstream in the dataset load.)")

    for split_name, records in [("train", train_records), ("dev", dev_records)]:
        dist = defaultdict(int)
        for r in records:
            dist[r["difficulty"]] += 1
        print(f"{split_name} difficulty distribution: {dict(dist)}")

    print(f"\nWrote {len(train_records)} train records and {len(dev_records)} dev records to {out_dir}/")


if __name__ == "__main__":
    main()

