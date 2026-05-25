"""Find an ungated Llama-class model. Try community mirrors that don't gate."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.model_utils import load_model

candidates = [
    "NousResearch/Meta-Llama-3-8B",
    "NousResearch/Llama-2-7b-hf",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "huggyllama/llama-7b",
    "openlm-research/open_llama_7b_v2",
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "unsloth/llama-3-8b",
]

succeeded = []
for name in candidates:
    try:
        print(f"\nTrying {name} ...", flush=True)
        model, tok = load_model(name, dtype=torch.bfloat16)
        cfg = model.config
        n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        d_head = cfg.hidden_size // cfg.num_attention_heads
        print(f"  SUCCESS: n_layers={cfg.num_hidden_layers}, n_kv_heads={n_kv}, d_head={d_head}", flush=True)
        print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
        succeeded.append((name, cfg.num_hidden_layers, n_kv, d_head))
        # free GPU
        del model
        torch.cuda.empty_cache()
        # stop after first success that has d_head divisible by 4 (HQMQ requires this)
        if d_head % 4 == 0:
            break
    except Exception as e:
        msg = str(e)[:250].replace("\n", " ")
        print(f"  FAILED: {msg}", flush=True)

print(f"\nSucceeded: {succeeded}")
