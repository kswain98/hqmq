"""Probe gpt-oss-20b loading and run a smoke-test perplexity sweep."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model_utils import load_model

candidates = [
    "openai/gpt-oss-20b",
    "unsloth/gpt-oss-20b",
    "openai/gpt-oss-120b",  # probably won't fit but check
]

for name in candidates:
    try:
        print(f"\nTrying {name} ...", flush=True)
        model, tok = load_model(name, dtype=torch.bfloat16)
        cfg = model.config
        n_kv = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", None))
        n_attn = getattr(cfg, "num_attention_heads", None)
        hidden = getattr(cfg, "hidden_size", None)
        d_head = hidden // n_attn if (hidden and n_attn) else None
        n_layers = getattr(cfg, "num_hidden_layers", None)
        print(f"  SUCCESS: n_layers={n_layers}, n_kv_heads={n_kv}, d_head={d_head}", flush=True)
        print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
        print(f"  config class: {type(cfg).__name__}", flush=True)
        # try a tiny forward
        inputs = tok("hello world", return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
            print(f"  forward OK, logits shape: {out.logits.shape}", flush=True)
        del model
        torch.cuda.empty_cache()
        break
    except Exception as e:
        msg = str(e)[:300].replace("\n", " ")
        print(f"  FAILED: {msg}", flush=True)
        torch.cuda.empty_cache()
