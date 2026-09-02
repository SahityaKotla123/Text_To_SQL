"""
Day 5 (part 1): Error analysis on the fine-tuned model's failures.

Categorizes failed predictions into rough buckets (wrong table, wrong join,
wrong aggregation, invalid SQL, etc.) using cheap structural heuristics
against the gold SQL — this won't be perfectly precise, but it's enough to
manually spot-check and describe patterns for your writeup. Read through
a sample of each bucket yourself; don't just report the counts blindly.

Usage:
    python src/error_analysis.py --results_path outputs/finetuned_results.json
"""

import argparse
import json
import re
from collections import defaultdict


def extract_tables(sql: str):
    return set(re.findall(r"\bFROM\s+(\w+)|\bJOIN\s+(\w+)", sql, flags=re.IGNORECASE)
               and [t for pair in re.findall(r"\bFROM\s+(\w+)|\bJOIN\s+(\w+)", sql, flags=re.IGNORECASE)
                    for t in pair if t])


def has_aggregation(sql: str):
    return bool(re.search(r"\bCOUNT\s*\(|\bSUM\s*\(|\bAVG\s*\(|\bMIN\s*\(|\bMAX\s*\(", sql, flags=re.IGNORECASE))


def has_group_by(sql: str):
    return "group by" in sql.lower()


def has_join(sql: str):
    return "join" in sql.lower() or sql.lower().count("from") and "," in sql.lower().split("from")[-1].split("where")[0]


def categorize_failure(pred_sql: str, gold_sql: str, valid_sql: bool):
    if not valid_sql:
        return "invalid_sql"

    pred_tables = extract_tables(pred_sql)
    gold_tables = extract_tables(gold_sql)
    if pred_tables != gold_tables:
        return "wrong_table_selection"

    if has_join(gold_sql) and not has_join(pred_sql):
        return "missing_join"
    if has_join(pred_sql) and not has_join(gold_sql):
        return "spurious_join"

    if has_aggregation(gold_sql) and not has_aggregation(pred_sql):
        return "missing_aggregation"
    if has_aggregation(pred_sql) and not has_aggregation(gold_sql):
        return "spurious_aggregation"

    if has_group_by(gold_sql) != has_group_by(pred_sql):
        return "group_by_mismatch"

    return "other_logic_error"  # right tables/structure, still wrong result — inspect manually


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default="outputs/finetuned_results.json")
    parser.add_argument("--out_path", type=str, default="outputs/error_categories.json")
    parser.add_argument("--samples_per_category", type=int, default=3)
    args = parser.parse_args()

    with open(args.results_path) as f:
        results = json.load(f)

    failures = [e for e in results["per_example"] if not e["execution_match"]]
    print(f"{len(failures)} / {results['n']} predictions failed execution match")

    categories = defaultdict(list)
    for e in failures:
        cat = categorize_failure(e["pred_sql"], e["gold_sql"], e["valid_sql"])
        categories[cat].append(e)

    print("\n=== FAILURE CATEGORY BREAKDOWN ===")
    summary = {}
    for cat, cases in sorted(categories.items(), key=lambda kv: -len(kv[1])):
        pct = 100 * len(cases) / max(len(failures), 1)
        print(f"  {cat:<25} {len(cases):4d}  ({pct:.1f}% of failures)")
        summary[cat] = {
            "count": len(cases),
            "pct_of_failures": pct,
            "examples": [
                {"question": c["question"], "gold_sql": c["gold_sql"], "pred_sql": c["pred_sql"],
                 "difficulty": c["difficulty"]}
                for c in cases[:args.samples_per_category]
            ],
        }

    with open(args.out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved category breakdown + sample cases to {args.out_path}")
    print("Read through the sampled cases in each category by hand before writing")
    print("your error-analysis section — the heuristic labels are a starting point,")
    print("not a substitute for actually looking at the SQL.")


if __name__ == "__main__":
    main()

