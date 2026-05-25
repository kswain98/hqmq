"""Find which Llama-class 7-8B model is downloadable from HF without gating."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.model_utils import load_model

candidates = [
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-2-7b-hf",
    "Qwen/Qwen2.5-7B",
    "openlm-research/open_llama_7b_v2",
    "tiiuae/falcon-7b",
]

for name in candidates:
    try:
        print(f"\nTrying {name} ...")
        model, tok = load_model(name, dtype=torch.bfloat16)
        cfg = model.config
        n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        d_head = cfg.hidden_size // cfg.num_attention_heads
        print(f"  SUCCESS: n_layers={cfg.num_hidden_layers}, n_kv_heads={n_kv}, d_head={d_head}")
        print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        sys.exit(0)
    except Exception as e:
        msg = str(e)[:250]
        print(f"  FAILED: {msg}")

print("\nNo candidates worked. Falling back to Mistral-7B-v0.1 already loaded.")
