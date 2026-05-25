"""Find an accessible Gemma mirror with d_head divisible by 4."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from huggingface_hub import hf_hub_download, HfApi

candidates = [
    "unsloth/gemma-2-9b",
    "unsloth/gemma-2-9b-it",
    "unsloth/gemma-2-2b",
    "unsloth/gemma-2-2b-it",
    "unsloth/gemma-3-4b-pt",
    "unsloth/gemma-3-12b-pt",
    "unsloth/gemma-3-4b-it",
    "unsloth/gemma-3-12b-it",
    "google/gemma-2-2b-it",  # try direct again
    "google/gemma-2-9b-it",
]

api = HfApi()
for name in candidates:
    try:
        info = api.model_info(name)
        gated = getattr(info, "gated", False)
        if gated and gated != False:
            print(f"{name}  GATED ({gated})")
            continue
        # Try downloading the config
        path = hf_hub_download(repo_id=name, filename="config.json")
        with open(path) as f:
            cfg = json.load(f)
        h = cfg.get("hidden_size") or cfg.get("text_config", {}).get("hidden_size")
        na = cfg.get("num_attention_heads") or cfg.get("text_config", {}).get("num_attention_heads")
        nkv = cfg.get("num_key_value_heads") or cfg.get("text_config", {}).get("num_key_value_heads", na)
        nl = cfg.get("num_hidden_layers") or cfg.get("text_config", {}).get("num_hidden_layers")
        head_dim = cfg.get("head_dim") or cfg.get("text_config", {}).get("head_dim")
        d_head = head_dim if head_dim else (h // na if (h and na) else None)
        ok = d_head and d_head % 4 == 0
        arch = cfg.get("model_type", cfg.get("text_config", {}).get("model_type", "?"))
        print(f"{name:34s} OPEN  {arch:14s} L={nl} n_attn={na} n_kv={nkv} d_head={d_head} HQMQ-ok={ok}")
    except Exception as e:
        print(f"{name:34s} ERROR: {str(e)[:100]}")
