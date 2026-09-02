"""
Day 6-7: Benchmark an OpenAI-compatible completion endpoint (works for
vLLM's api_server, and TensorRT-LLM/Triton if fronted with an
OpenAI-compatible layer). Measures TTFT, tokens/sec, and p50/p95 latency
over a sample of dev prompts.

Usage:
    python src/benchmark_serving.py \
        --endpoint http://localhost:8000/v1/completions \
        --model_name outputs/merged_model_awq \
        --dev_path data/dev.jsonl --n_requests 50
"""

import argparse
import json
import time

import numpy as np
import requests


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def benchmark(endpoint, model_name, prompts, max_tokens=256):
    ttfts = []
    total_latencies = []
    token_counts = []

    for prompt in prompts:
        start = time.perf_counter()
        resp = requests.post(endpoint, json={
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }, timeout=60)
        end = time.perf_counter()
        resp.raise_for_status()
        data = resp.json()

        total_latency = end - start
        total_latencies.append(total_latency)

        completion = data["choices"][0]["text"]
        approx_tokens = max(len(completion.split()), 1)
        token_counts.append(approx_tokens)

        # Non-streaming request: TTFT approximated as total latency here.
        # For a real TTFT measurement, use stream=True and time the first
        # chunk instead — left as a follow-up if you want a sharper number.
        ttfts.append(total_latency)

    return {
        "n_requests": len(prompts),
        "ttft_p50_s": float(np.percentile(ttfts, 50)),
        "ttft_p95_s": float(np.percentile(ttfts, 95)),
        "latency_p50_s": float(np.percentile(total_latencies, 50)),
        "latency_p95_s": float(np.percentile(total_latencies, 95)),
        "avg_tokens_per_request": float(np.mean(token_counts)),
        "approx_tokens_per_sec": float(
            sum(token_counts) / sum(total_latencies)
        ) if sum(total_latencies) > 0 else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dev_path", type=str, default="data/dev.jsonl")
    parser.add_argument("--schema_variant", type=str, default="rich")
    parser.add_argument("--n_requests", type=int, default=50)
    parser.add_argument("--out_path", type=str, default="outputs/serving_benchmark.json")
    args = parser.parse_args()

    dev_records = load_jsonl(args.dev_path)[:args.n_requests]
    prompts = [r[f"prompt_{args.schema_variant}"] for r in dev_records]

    print(f"Benchmarking {args.endpoint} with {len(prompts)} requests...")
    results = benchmark(args.endpoint, args.model_name, prompts)

    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SERVING BENCHMARK (model-level) ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {args.out_path}")
    print("Remember: compare accuracy on THESE outputs against Day 4's unquantized")
    print("fine-tuned accuracy to report accuracy retention after quantization.")


if __name__ == "__main__":
    main()

