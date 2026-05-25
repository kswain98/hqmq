"""Needle-in-a-Haystack retrieval test.

Tests whether the model can retrieve a specific fact embedded in a long context,
with the KV cache passed through HQMQ. This is the canonical real-world eval
for KV cache quantization: the model needs to attend over many tokens to find
the needle, and quantization noise can disrupt the routing.

Setup per trial:
  1. Take a long base passage (random text from WikiText or PG-19).
  2. Insert a needle sentence at a target depth (e.g., 30%, 50%, 70%).
  3. Append a query: "What is the magic number?"
  4. Generate a short completion with QuantizedCache.
  5. Score: does the needle answer appear in the output?
"""

import random
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers import DynamicCache

from ..model_utils import QuantizedCache
from ..quantizers.base import KVQuantizer


def _load_haystack_text(min_tokens: int = 30000) -> str:
    """Load a chunk of long text from WikiText-103 train for use as filler."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    chunks, total = [], 0
    for ex in ds:
        if not ex["text"].strip():
            continue
        chunks.append(ex["text"])
        total += len(ex["text"].split())
        if total >= min_tokens:
            break
    return "\n".join(chunks)


@torch.no_grad()
def needle_in_haystack(
    model, tokenizer,
    quantizer: Optional[KVQuantizer],
    *,
    seq_len: int = 4096,
    depths: List[float] = (0.25, 0.5, 0.75),
    n_trials_per_depth: int = 4,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """Insert a needle at varying depths in a long context and test retrieval.

    For each trial: pick a random 4-digit number, embed in context at the given depth,
    ask the model to retrieve, score by exact-match.
    """
    rng = random.Random(seed)
    haystack_text = _load_haystack_text(min_tokens=seq_len * 4)
    haystack_tokens = tokenizer(haystack_text, return_tensors="pt").input_ids[0]
    if haystack_tokens.size(0) < seq_len:
        raise RuntimeError(f"Haystack too short: {haystack_tokens.size(0)} < {seq_len}")

    results = []
    n_correct, n_total = 0, 0
    for depth in depths:
        for trial in range(n_trials_per_depth):
            # Random 4-digit magic number
            magic = rng.randint(1000, 9999)
            needle = f"\nThe magic number is {magic}.\n"
            query = "\n\nQuestion: What is the magic number?\nAnswer:"

            # Carve a context to fit (haystack + needle + query) into seq_len
            needle_ids = tokenizer(needle, return_tensors="pt").input_ids[0]
            query_ids = tokenizer(query, return_tensors="pt").input_ids[0]
            n_filler = seq_len - needle_ids.size(0) - query_ids.size(0)
            assert n_filler > 100, f"seq_len={seq_len} too short for needle+query"

            start = rng.randint(0, haystack_tokens.size(0) - n_filler - 1)
            filler = haystack_tokens[start : start + n_filler]
            depth_pos = int(depth * n_filler)
            prefix = filler[:depth_pos]
            suffix = filler[depth_pos:]
            full_ids = torch.cat([prefix, needle_ids, suffix, query_ids]).unsqueeze(0).to(device)

            # Generate ~10 tokens with QuantizedCache
            cache = QuantizedCache(quantizer) if quantizer is not None else DynamicCache()
            with torch.no_grad():
                out_ids = model.generate(
                    full_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    past_key_values=cache,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out_ids[0, full_ids.size(1):]
            answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            correct = str(magic) in answer_text
            n_correct += int(correct)
            n_total += 1
            results.append({"depth": depth, "trial": trial, "magic": magic,
                            "answer_text": answer_text, "correct": correct})

    return {
        "task": "needle_in_haystack",
        "acc": n_correct / n_total,
        "n": n_total,
        "seq_len": seq_len,
        "quantizer": quantizer.name if quantizer is not None else "fp16_baseline",
        "bits_per_value": quantizer.bits_per_value() if quantizer is not None else 16.0,
        "per_trial": results,
    }
