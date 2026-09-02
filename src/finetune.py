"""
Day 3: QLoRA fine-tune via Unsloth.

Run this in Colab. Unsloth pins to specific torch/CUDA versions, so install
it via their official Colab install cell rather than requirements.txt:

    !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

Usage:
    python src/finetune.py \
        --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
        --train_path data/train.jsonl \
        --schema_variant rich \
        --output_dir outputs/qlora_adapter
"""

import argparse
import json

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--train_path", type=str, default="data/train.jsonl")
    parser.add_argument("--schema_variant", type=str, choices=["minimal", "rich"], default="rich")
    parser.add_argument("--output_dir", type=str, default="outputs/qlora_adapter")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading base model in 4-bit: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,          # auto-detect
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,      # 0 is optimized in Unsloth
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    train_records = load_jsonl(args.train_path)
    prompt_key = f"prompt_{args.schema_variant}"

    def to_text(rec):
        # Chat-formatted, matching generate.py's apply_chat_formatting exactly:
        # user turn = the schema+question prompt, assistant turn = gold SQL.
        # This has to match what's used at eval time (see generate.py) or the
        # fine-tuned model is learning one format and being scored on another.
        messages = [
            {"role": "user", "content": rec[prompt_key]},
            {"role": "assistant", "content": rec["gold_sql"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    train_dataset = Dataset.from_list([to_text(r) for r in train_records])

    print(f"Training on {len(train_dataset)} examples "
          f"(schema_variant={args.schema_variant}) for {args.epochs} epoch(s)...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.05,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=not torch_bf16_supported(),
            bf16=torch_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=args.seed,
            output_dir=args.output_dir,
            save_strategy="epoch",
            report_to="none",
        ),
    )

    trainer.train()

    print(f"Saving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Record the config used — Phase 2's freeze-before-test discipline
    # starts here: write down exactly what produced this checkpoint.
    with open(f"{args.output_dir}/run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print("Wrote run_config.json alongside the adapter for reproducibility.")


def torch_bf16_supported():
    import torch
    return torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False


if __name__ == "__main__":
    main()

