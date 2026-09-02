"""
Day 5 (part 2): Schema serialization ablation.

Compares two evaluation runs of the SAME fine-tuned model — one run with
--schema_variant minimal, one with --schema_variant rich (types, PKs, FKs) —
per the review feedback that "schema present vs absent" is a strawman, and
"minimal vs rich serialization" is the real design question.

Run evaluate_finetuned.py twice first (same adapter, different schema_variant,
different --out_path), then point this script at both result files:

    python src/evaluate_finetuned.py ... --schema_variant minimal --out_path outputs/ft_minimal.json
    python src/evaluate_finetuned.py ... --schema_variant rich    --out_path outputs/ft_rich.json
    python src/ablation_compare.py --minimal_path outputs/ft_minimal.json --rich_path outputs/ft_rich.json
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimal_path", type=str, default="outputs/ft_minimal.json")
    parser.add_argument("--rich_path", type=str, default="outputs/ft_rich.json")
    args = parser.parse_args()

    with open(args.minimal_path) as f:
        minimal = json.load(f)
    with open(args.rich_path) as f:
        rich = json.load(f)

    print("=== SCHEMA SERIALIZATION ABLATION ===")
    print(f"{'Variant':<12} {'ExecAcc':>10} {'95% CI':>18} {'ExactMatch':>12} {'ValidSQL':>10}")
    for name, res in [("Minimal", minimal), ("Rich", rich)]:
        ci = res["execution_accuracy_ci95"]
        print(f"{name:<12} {res['execution_accuracy']:>10.3f} "
              f"[{ci[0]:.3f}, {ci[1]:.3f}]   {res['exact_match']:>10.3f}   {res['valid_sql_rate']:>8.3f}")

    print("\nBy difficulty (minimal -> rich):")
    for diff in sorted(set(minimal["by_difficulty"]) | set(rich["by_difficulty"])):
        m = minimal["by_difficulty"].get(diff, {"execution_accuracy": float("nan"), "n": 0})
        r = rich["by_difficulty"].get(diff, {"execution_accuracy": float("nan"), "n": 0})
        print(f"  {diff:12s} n={r['n']:4d}  {m['execution_accuracy']:.3f} -> {r['execution_accuracy']:.3f}")

    delta = rich["execution_accuracy"] - minimal["execution_accuracy"]
    print(f"\nOverall execution accuracy delta (rich - minimal): {delta:+.3f}")
    print("Interpretation guide:")
    print("  - Large positive delta on hard/extra_hard queries -> model relies on")
    print("    types/FKs mainly for complex joins and multi-table reasoning.")
    print("  - Small/no delta -> table+column names alone may be sufficient signal")
    print("    for this model/task, which is itself a useful (if less flashy) finding.")


if __name__ == "__main__":
    main()

