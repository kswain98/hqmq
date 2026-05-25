"""Probe modern open models for HQMQ compatibility (d_head divisible by 4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from huggingface_hub import HfApi

candidates = [
    "google/gemma-2-9b",
    "google/gemma-2-2b",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "deepseek-ai/DeepSeek-V2-Lite",
    "meta-llama/Llama-3.1-8B",
    "NousResearch/Hermes-3-Llama-3.1-8B",
]

api = HfApi()
for name in candidates:
    try:
        info = api.model_info(name)
        gated = getattr(info, "gated", False)
        config = info.config if hasattr(info, "config") else {}
        # config has hidden_size, num_attention_heads, num_hidden_layers
        h = config.get("hidden_size")
        na = config.get("num_attention_heads")
        nkv = config.get("num_key_value_heads", na)
        nl = config.get("num_hidden_layers")
        d_head = h // na if (h and na) else None
        ok = d_head and d_head % 4 == 0
        print(f"{name:50s}  gated={str(gated):8s}  layers={nl}  n_attn={na}  n_kv={nkv}  d_head={d_head}  HQMQ-ok={ok}")
    except Exception as e:
        print(f"{name:50s}  ERROR: {str(e)[:80]}")
