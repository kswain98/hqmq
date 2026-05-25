"""Model loading + QuantizedCache wrapper.

`QuantizedCache` is a drop-in replacement for HuggingFace's DynamicCache
that applies a KVQuantizer to every K, V tensor as it's appended. This is
"fake quantization": stored values are dequantized fp16, but they've passed
through the bit-width-limiting quantize/dequantize step.

Usage:
    model, tokenizer = load_model("EleutherAI/pythia-1b-deduped")
    cache = QuantizedCache(NaivePerTokenIntQuantizer(bits=4))
    out = model(input_ids, past_key_values=cache, use_cache=True)
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .quantizers.base import KVQuantizer


class QuantizedCache(DynamicCache):
    """DynamicCache that quantizes K, V on every update."""

    def __init__(self, quantizer: KVQuantizer):
        super().__init__()
        self.quantizer = quantizer

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k_q = self.quantizer.quantize_K(key_states, layer_idx)
        v_q = self.quantizer.quantize_V(value_states, layer_idx)
        return super().update(k_q, v_q, layer_idx, cache_kwargs)


def load_model(model_name: str, dtype: torch.dtype = torch.float16, device: str = "cuda"):
    """Load an HF causal LM + tokenizer. Returns (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation="eager"
    ).to(device)
    model.eval()
    return model, tokenizer
