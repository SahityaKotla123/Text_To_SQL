"""
Shared generation utility. Used by baseline_eval.py, finetune.py's post-train
eval, and the schema ablation script — one code path so results are
comparable across all of them.
"""

import re
import torch
from tqdm import tqdm


def extract_sql(raw_output: str) -> str:
    """
    Model output may include markdown fences, trailing commentary, etc.
    Pull out just the SQL statement.
    """
    text = raw_output.strip()
    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    # If there's a stray "### " section marker leaking through, cut at it
    text = text.split("###")[0].strip()
    # Keep only up to the first semicolon-terminated statement if multiple present
    if ";" in text:
        text = text.split(";")[0].strip() + ";"
    return text


def apply_chat_formatting(tokenizer, prompts: list) -> list:
    """
    Wraps each raw prompt as a single user turn and renders it through the
    model's chat template. Qwen2.5-Coder-Instruct (like most Instruct models)
    is fine-tuned to expect this format — feeding it a raw completion-style
    prompt instead makes it ramble/over-explain and produces far more
    malformed SQL than it should. This is applied identically at baseline
    eval, fine-tuned eval, and training time (see finetune.py) so the
    comparison stays apples-to-apples.
    """
    formatted = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted.append(text)
    return formatted


@torch.no_grad()
def generate_predictions(model, tokenizer, prompts: list, batch_size: int = 8,
                          max_new_tokens: int = 256, device: str = "cuda",
                          chat_format: bool = True):
    """
    Batched greedy generation. Greedy (do_sample=False) is intentional here —
    we want a deterministic, reproducible eval, not sampling variance muddying
    the before/after comparison.

    chat_format=True (default) renders prompts through the tokenizer's chat
    template before generation — required for Instruct models to behave
    correctly. Only set False if you're using a base (non-Instruct) model
    that was prompted/trained in raw completion style.
    """
    model.eval()
    if chat_format:
        prompts = apply_chat_formatting(tokenizer, prompts)
    predictions = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="generating"):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=2048).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        for j in range(len(batch)):
            input_len = inputs["input_ids"][j].shape[0]
            gen_tokens = outputs[j][input_len:]
            raw = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            predictions.append(extract_sql(raw))

    return predictions

