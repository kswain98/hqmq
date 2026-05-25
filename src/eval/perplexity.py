"""WikiText-103 perplexity with quantized KV cache.

Standard sliding-window evaluation. Each window is run with a fresh
QuantizedCache so the quantization effects compound as the cache grows.
Loss is computed over the whole window (token i predicts token i+1).
"""

from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import DynamicCache

from ..model_utils import QuantizedCache
from ..quantizers.base import KVQuantizer


@torch.no_grad()
def wikitext_perplexity(
    model,
    tokenizer,
    quantizer: Optional[KVQuantizer] = None,
    *,
    seq_len: int = 2048,
    stride: int = 2048,
    max_windows: Optional[int] = None,
    device: str = "cuda",
) -> dict:
    """Compute mean perplexity on WikiText-103 test split.

    If quantizer is None, uses standard fp16 DynamicCache (true baseline).
    """
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    n_tokens = input_ids.size(1)

    nlls = []
    n_processed = 0
    starts = list(range(0, n_tokens - seq_len, stride))
    if max_windows is not None:
        starts = starts[:max_windows]

    for i, begin in enumerate(starts):
        end = begin + seq_len
        window = input_ids[:, begin:end]
        cache = QuantizedCache(quantizer) if quantizer is not None else DynamicCache()
        out = model(window, past_key_values=cache, use_cache=True, labels=window)
        # out.loss is mean cross-entropy over (seq_len - 1) positions
        nll = out.loss.item() * (seq_len - 1)
        nlls.append(nll)
        n_processed += (seq_len - 1)

    total_nll = sum(nlls)
    ppl = float(torch.exp(torch.tensor(total_nll / n_processed)))
    return {
        "perplexity": ppl,
        "n_windows": len(starts),
        "n_tokens": n_processed,
        "quantizer": quantizer.name if quantizer is not None else "fp16_baseline",
        "bits_per_value": quantizer.bits_per_value() if quantizer is not None else 16.0,
    }
