"""
Day 4: Evaluate the fine-tuned model on the SAME dev set, using the SAME
schema variant as baseline_eval.py, and build the comparison table.

Usage:
    python src/evaluate_finetuned.py \
        --base_model_name Qwen/Qwen2.5-Coder-7B-Instruct \
        --adapter_path outputs/qlora_adapter \
        --databases_dir /path/to/spider/database \
        --schema_variant rich \
        --baseline_results outputs/baseline_results.json
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from generate import generate_predictions
from eval_utils import evaluate_predictions, bootstrap_ci


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_regression_analysis(baseline_results, finetuned_results):
    """
    Per review feedback: don't just report average accuracy delta — check
    where the base model was RIGHT and the fine-tuned model got it WRONG.
    A net accuracy gain can hide real regressions on a subset of queries.
    """
    base_by_q = {(e["db_id"], e["gold_sql"]): e for e in baseline_results["per_example"]}
    regressions = []
    improvements = []
    for e in finetuned_results["per_example"]:
        key = (e["db_id"], e["gold_sql"])
        base_e = base_by_q.get(key)
        if base_e is None:
            continue
        if base_e["execution_match"] and not e["execution_match"]:
            regressions.append({
                "db_id": e["db_id"], "question": e["question"],
                "gold_sql": e["gold_sql"],
                "base_pred": base_e["pred_sql"], "finetuned_pred": e["pred_sql"],
                "difficulty": e["difficulty"],
            })
        elif not base_e["execution_match"] and e["execution_match"]:
            improvements.append({
                "db_id": e["db_id"], "question": e["question"],
                "gold_sql": e["gold_sql"],
                "base_pred": base_e["pred_sql"], "finetuned_pred": e["pred_sql"],
                "difficulty": e["difficulty"],
            })
    return regressions, improvements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="outputs/qlora_adapter")
    parser.add_argument("--dev_path", type=str, default="data/dev.jsonl")
    parser.add_argument("--databases_dir", type=str, required=True)
    parser.add_argument("--schema_variant", type=str, choices=["minimal", "rich"], default="rich")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--baseline_results", type=str, default="outputs/baseline_results.json")
    parser.add_argument("--out_path", type=str, default="outputs/finetuned_results.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                         help="Load base model in 4-bit before attaching the LoRA adapter "
                              "(default: on). This matches how the adapter was trained "
                              "(QLoRA) and avoids CPU offload slowness on a T4.")
    parser.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")
    args = parser.parse_args()

    dev_records = load_jsonl(args.dev_path)
    if args.limit:
        dev_records = dev_records[:args.limit]
    prompt_key = f"prompt_{args.schema_variant}"
    prompts = [r[prompt_key] for r in dev_records]

    print(f"Loading base model {args.base_model_name} (4-bit={args.load_in_4bit}) + adapter {args.adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name, quantization_config=bnb_config, device_map="auto"
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name, torch_dtype="auto", device_map="auto"
        )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print(f"Generating predictions for {len(prompts)} dev examples...")
    predictions = generate_predictions(model, tokenizer, prompts, batch_size=args.batch_size)

    print("Scoring...")
    finetuned_results = evaluate_predictions(dev_records, predictions, Path(args.databases_dir))

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(finetuned_results, f, indent=2)

    print("\n=== FINE-TUNED MODEL RESULTS ===")
    print(f"Execution accuracy: {finetuned_results['execution_accuracy']:.3f} "
          f"(95% CI: {finetuned_results['execution_accuracy_ci95'][0]:.3f}-"
          f"{finetuned_results['execution_accuracy_ci95'][1]:.3f})")
    print(f"Exact match:        {finetuned_results['exact_match']:.3f}")
    print(f"Valid SQL rate:     {finetuned_results['valid_sql_rate']:.3f}")

    if Path(args.baseline_results).exists():
        with open(args.baseline_results) as f:
            baseline_results = json.load(f)

        print("\n=== COMPARISON TABLE ===")
        print(f"{'Model':<20} {'ExecAcc':>10} {'95% CI':>18} {'ExactMatch':>12} {'ValidSQL':>10}")
        for name, res in [("Base", baseline_results), ("Fine-tuned", finetuned_results)]:
            ci = res["execution_accuracy_ci95"]
            print(f"{name:<20} {res['execution_accuracy']:>10.3f} "
                  f"[{ci[0]:.3f}, {ci[1]:.3f}]   "
                  f"{res['exact_match']:>10.3f}   {res['valid_sql_rate']:>8.3f}")

        print("\nBy difficulty (base -> fine-tuned):")
        for diff in sorted(set(baseline_results["by_difficulty"]) | set(finetuned_results["by_difficulty"])):
            b = baseline_results["by_difficulty"].get(diff, {"execution_accuracy": float("nan"), "n": 0})
            ft = finetuned_results["by_difficulty"].get(diff, {"execution_accuracy": float("nan"), "n": 0})
            print(f"  {diff:12s} n={ft['n']:4d}  {b['execution_accuracy']:.3f} -> {ft['execution_accuracy']:.3f}")

        print("\nRunning regression analysis (base correct -> fine-tuned wrong)...")
        regressions, improvements = build_regression_analysis(baseline_results, finetuned_results)
        print(f"  Regressions:  {len(regressions)} cases (base was right, fine-tuned got it wrong)")
        print(f"  Improvements: {len(improvements)} cases (base was wrong, fine-tuned got it right)")

        with open("outputs/regression_analysis.json", "w") as f:
            json.dump({"regressions": regressions, "improvements": improvements}, f, indent=2)
        print("  Saved full case lists to outputs/regression_analysis.json")
        print("  -> Read through the regressions in Day 5's error analysis. If there are many,")
        print("     that's a real finding to report, not something to bury.")
    else:
        print(f"\nNo baseline results found at {args.baseline_results} — run baseline_eval.py first "
              "to get the comparison table.")


if __name__ == "__main__":
    main()

