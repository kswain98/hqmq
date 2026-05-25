"""Long-context perplexity evaluation.

Same idea as wikitext_perplexity but with longer sequences (4k, 8k, 16k tokens)
to evaluate KV quantization quality in its actual deployment regime. Uses
PG-19 or WikiText-103 long passages.
"""

from typing import Optional

import torch
from datasets import load_dataset
from transformers import DynamicCache

from ..model_utils import QuantizedCache
from ..quantizers.base import KVQuantizer


@torch.no_grad()
def long_context_perplexity(
    model,
    tokenizer,
    quantizer: Optional[KVQuantizer] = None,
    *,
    seq_len: int = 4096,
    max_windows: int = 5,
    dataset: str = "wikitext-103",
    device: str = "cuda",
) -> dict:
    """Compute perplexity over longer non-overlapping windows.

    For wikitext-103: concatenates all test passages and slices into seq_len chunks.
    For pg19: uses individual long documents.
    """
    if dataset == "wikitext-103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    elif dataset == "pg19":
        ds = load_dataset("deepmind/pg19", split="test", streaming=True)
        # Take first few long docs
        texts = []
        for i, ex in enumerate(ds):
            if i >= max_windows:
                break
            texts.append(ex["text"])
        text = "\n\n".join(texts)
    else:
        raise ValueError(dataset)

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    n_tokens = input_ids.size(1)
    starts = list(range(0, n_tokens - seq_len, seq_len))[:max_windows]

    nlls, n_pred = [], 0
    for begin in starts:
        end = begin + seq_len
        window = input_ids[:, begin:end]
        cache = QuantizedCache(quantizer) if quantizer is not None else DynamicCache()
        out = model(window, past_key_values=cache, use_cache=True, labels=window)
        nlls.append(out.loss.item() * (seq_len - 1))
        n_pred += seq_len - 1

    ppl = float(torch.exp(torch.tensor(sum(nlls) / n_pred)))
    return {
        "perplexity": ppl,
        "n_windows": len(starts),
        "n_tokens": n_pred,
        "seq_len": seq_len,
        "quantizer": quantizer.name if quantizer is not None else "fp16_baseline",
        "bits_per_value": quantizer.bits_per_value() if quantizer is not None else 16.0,
    }
