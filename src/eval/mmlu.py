"""MMLU 5-shot evaluation under QuantizedCache.

MMLU has 57 subjects; we use cais/mmlu via HuggingFace datasets. For each
test example, we form a 5-shot prompt from the matching subject's "dev" split
and score the 4 multiple-choice options (A/B/C/D) by single-token likelihood.

We use single-token scoring of the answer letter (A/B/C/D), which is the
standard MMLU evaluation protocol (matches lm-eval-harness).
"""

from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset

from ..model_utils import QuantizedCache
from ..quantizers.base import KVQuantizer


@torch.no_grad()
def score_answer_letters(
    model, tokenizer, prompt: str, quantizer: Optional[KVQuantizer], device: str = "cuda",
    max_len: int = 4096,
):
    """Return log-prob for each of "A", "B", "C", "D" as the next token after prompt."""
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
    # Use QuantizedCache so quantization is applied during the forward
    from transformers import DynamicCache
    if quantizer is None:
        cache = DynamicCache()
    else:
        cache = QuantizedCache(quantizer=quantizer)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    logits = out.logits[0, -1, :].float().detach()
    log_probs = F.log_softmax(logits, dim=-1)

    letter_token_ids = []
    for letter in ["A", "B", "C", "D"]:
        # Tokenize with leading space (post-newline context standard for MMLU)
        tok = tokenizer(letter, add_special_tokens=False).input_ids
        # Tokenizer may produce 1-2 tokens depending on model; take last as the letter
        letter_token_ids.append(tok[-1] if tok else 0)

    result = [log_probs[t].item() for t in letter_token_ids]
    # Clear cache and logits explicitly
    del cache, out, logits, log_probs, ids
    return result


def format_mmlu_example(question, choices, answer_letter=None, with_answer=False):
    """Format a single MMLU example. choices is 4 strings, answer_letter A/B/C/D or None."""
    s = f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
    if with_answer and answer_letter is not None:
        s += f" {answer_letter}"
    return s


@torch.no_grad()
def eval_mmlu_5shot(model, tokenizer, quantizer: Optional[KVQuantizer],
                    max_examples: int = 500, device: str = "cuda", n_shots: int = 5,
                    subjects: Optional[list] = None):
    """5-shot MMLU evaluation. Returns {acc, n, per_subject}.

    By default samples ~500 examples uniformly across MMLU's 57 subjects.
    """
    # Load all subjects (cais/mmlu config "all")
    test_ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    dev_ds = load_dataset("cais/mmlu", "all", split="dev", trust_remote_code=True)

    # Group dev by subject for n_shot prompt construction
    dev_by_subject = {}
    for ex in dev_ds:
        dev_by_subject.setdefault(ex["subject"], []).append(ex)

    # Subset test examples
    if subjects:
        test_ds = test_ds.filter(lambda x: x["subject"] in subjects)
    if len(test_ds) > max_examples:
        idxs = list(range(0, len(test_ds), max(1, len(test_ds) // max_examples)))[:max_examples]
        test_ds = test_ds.select(idxs)

    correct = 0
    total = 0
    per_subject = {}
    letters = ["A", "B", "C", "D"]

    for ex in test_ds:
        subj = ex["subject"]
        dev_exs = dev_by_subject.get(subj, [])
        prompt_parts = [f"The following are multiple choice questions about {subj.replace('_', ' ')}.\n"]
        for shot in dev_exs[:n_shots]:
            prompt_parts.append(
                format_mmlu_example(shot["question"], shot["choices"],
                                     letters[shot["answer"]], with_answer=True))
        prompt_parts.append(
            format_mmlu_example(ex["question"], ex["choices"], with_answer=False))
        full_prompt = "\n\n".join(prompt_parts)

        scores = score_answer_letters(model, tokenizer, full_prompt, quantizer, device)
        pred = max(range(4), key=lambda i: scores[i])
        gold = ex["answer"]  # int 0-3
        ok = (pred == gold)
        correct += int(ok)
        total += 1
        s = per_subject.setdefault(subj, {"correct": 0, "n": 0})
        s["correct"] += int(ok)
        s["n"] += 1
        if total % 50 == 0:
            torch.cuda.empty_cache()

    per_subject_acc = {k: v["correct"] / v["n"] for k, v in per_subject.items()}
    return {"task": "mmlu_5shot", "acc": correct / total if total else 0.0, "n": total,
            "per_subject": per_subject_acc}
