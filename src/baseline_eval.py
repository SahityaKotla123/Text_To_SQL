"""
Day 2 (part 1): Evaluate the UNTOUCHED base model on Spider's official dev
split before any fine-tuning happens. This is your honest baseline — run it
first, and don't be surprised if it's already decent (Qwen2.5-Coder may have
seen SQL-heavy pretraining data). Report it as-is.

Usage:
    python src/baseline_eval.py \
        --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
        --databases_dir /path/to/spider/database \
        --schema_variant rich
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from generate import generate_predictions
from eval_utils import evaluate_predictions


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--dev_path", type=str, default="data/dev.jsonl")
    parser.add_argument("--databases_dir", type=str, required=True)
    parser.add_argument("--schema_variant", type=str, choices=["minimal", "rich"], default="rich")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out_path", type=str, default="outputs/baseline_results.json")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optionally cap dev set size for a quick smoke test")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                         help="Load in 4-bit (default: on). Needed to fit a 7B model "
                              "comfortably on a T4 without CPU offload, which is extremely "
                              "slow. Also keeps this eval consistent with Day 3's QLoRA "
                              "fine-tuning, which trains on a 4-bit base model.")
    parser.add_argument("--no_4bit", dest="load_in_4bit", action="store_false",
                         help="Disable 4-bit loading (only if you have a bigger GPU, e.g. A100).")
    args = parser.parse_args()

    dev_records = load_jsonl(args.dev_path)
    if args.limit:
        dev_records = dev_records[:args.limit]

    prompt_key = f"prompt_{args.schema_variant}"
    prompts = [r[prompt_key] for r in dev_records]

    print(f"Loading base model: {args.model_name} (4-bit={args.load_in_4bit})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generation

    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, quantization_config=bnb_config, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype="auto", device_map="auto"
        )

    print(f"Generating predictions for {len(prompts)} dev examples "
          f"(schema_variant={args.schema_variant})...")
    predictions = generate_predictions(model, tokenizer, prompts, batch_size=args.batch_size)

    print("Scoring against Spider databases...")
    results = evaluate_predictions(dev_records, predictions, Path(args.databases_dir))

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== BASE MODEL RESULTS (untouched, pre-fine-tuning) ===")
    print(f"n = {results['n']}")
    print(f"Execution accuracy: {results['execution_accuracy']:.3f} "
          f"(95% CI: {results['execution_accuracy_ci95'][0]:.3f}-{results['execution_accuracy_ci95'][1]:.3f})")
    print(f"Exact match:        {results['exact_match']:.3f}")
    print(f"Valid SQL rate:     {results['valid_sql_rate']:.3f}")
    print("By difficulty:")
    for diff, stats in results["by_difficulty"].items():
        print(f"  {diff:12s} n={stats['n']:4d}  exec_acc={stats['execution_accuracy']:.3f}")
    print(f"\nSaved full results (including per-example predictions) to {args.out_path}")


if __name__ == "__main__":
    main()

