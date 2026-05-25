"""Minimal multiple-choice task eval with QuantizedCache.

Implements zero-shot scoring for PIQA, HellaSwag, ARC-easy. For each example,
compute mean log-likelihood of each candidate continuation under the model
with the given quantizer, then pick argmax. Compare to gold label for accuracy.

Avoids the lm-eval-harness dependency (not installed in this env).
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import DynamicCache

from ..model_utils import QuantizedCache
from ..quantizers.base import KVQuantizer


@torch.no_grad()
def score_continuations(
    model, tokenizer, prompt: str, continuations: List[str],
    quantizer: Optional[KVQuantizer], device: str = "cuda",
) -> List[float]:
    """For each continuation c_i, return mean log P(c_i tokens | prompt) under the model.

    Uses a fresh QuantizedCache per example so quantization noise accumulates as it would
    in a real generation. Returns mean log-prob (higher = more likely).
    """
    scores = []
    for cont in continuations:
        full_text = prompt + cont
        ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        n_prompt = prompt_ids.size(1)
        n_total = ids.size(1)
        n_cont = n_total - n_prompt
        if n_cont <= 0:
            scores.append(-float("inf"))
            continue
        cache = QuantizedCache(quantizer) if quantizer is not None else DynamicCache()
        out = model(ids, past_key_values=cache, use_cache=True)
        logits = out.logits  # (1, T, V)
        # Score the continuation tokens: predict token i from logits[i-1]
        # Continuation tokens are at positions [n_prompt, n_total)
        # Their predictions come from logits at positions [n_prompt-1, n_total-1)
        log_probs = F.log_softmax(logits[0, n_prompt - 1 : n_total - 1].float(), dim=-1)
        cont_tokens = ids[0, n_prompt:]  # (n_cont,)
        token_lls = log_probs.gather(-1, cont_tokens.unsqueeze(-1)).squeeze(-1)  # (n_cont,)
        scores.append(token_lls.mean().item())
    return scores


@torch.no_grad()
def eval_piqa(model, tokenizer, quantizer, *, max_examples: int = 500, device: str = "cuda") -> dict:
    ds = load_dataset("piqa", split="validation", trust_remote_code=True)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    n_correct = 0
    n_total = 0
    for ex in ds:
        prompt = f"Question: {ex['goal']}\nAnswer:"
        continuations = [" " + ex["sol1"], " " + ex["sol2"]]
        scores = score_continuations(model, tokenizer, prompt, continuations, quantizer, device)
        pred = int(scores[1] > scores[0])
        if pred == ex["label"]:
            n_correct += 1
        n_total += 1
    return {"task": "piqa", "acc": n_correct / n_total, "n": n_total}


@torch.no_grad()
def eval_hellaswag(model, tokenizer, quantizer, *, max_examples: int = 500, device: str = "cuda") -> dict:
    ds = load_dataset("hellaswag", split="validation", trust_remote_code=True)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    n_correct = 0
    n_total = 0
    for ex in ds:
        prompt = f"{ex['activity_label']}: {ex['ctx']}"
        continuations = [" " + e for e in ex["endings"]]
        scores = score_continuations(model, tokenizer, prompt, continuations, quantizer, device)
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        if str(pred) == ex["label"]:
            n_correct += 1
        n_total += 1
    return {"task": "hellaswag", "acc": n_correct / n_total, "n": n_total}


@torch.no_grad()
def eval_arc_easy(model, tokenizer, quantizer, *, max_examples: int = 500, device: str = "cuda") -> dict:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation", trust_remote_code=True)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    n_correct = 0
    n_total = 0
    for ex in ds:
        prompt = f"Question: {ex['question']}\nAnswer:"
        choice_texts = ex["choices"]["text"]
        choice_labels = ex["choices"]["label"]
        continuations = [" " + c for c in choice_texts]
        scores = score_continuations(model, tokenizer, prompt, continuations, quantizer, device)
        pred_idx = max(range(len(scores)), key=lambda i: scores[i])
        pred_label = choice_labels[pred_idx]
        if pred_label == ex["answerKey"]:
            n_correct += 1
        n_total += 1
    return {"task": "arc_easy", "acc": n_correct / n_total, "n": n_total}
