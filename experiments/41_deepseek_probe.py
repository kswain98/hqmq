"""Probe DeepSeek-V2-Lite KV cache structure.

Goal: see what shapes the cache write hooks see during a forward pass, so we
can decide if (and how) HQMQ can be adapted.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading DeepSeek-V2-Lite ...", flush=True)
model_id = "deepseek-ai/DeepSeek-V2-Lite"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda",
)
cfg = model.config
print(f"  model_type={cfg.model_type}")
print(f"  layers={cfg.num_hidden_layers}, n_attn={cfg.num_attention_heads}, n_kv={cfg.num_key_value_heads}")
print(f"  kv_lora_rank={getattr(cfg, 'kv_lora_rank', None)}")
print(f"  qk_nope_head_dim={getattr(cfg, 'qk_nope_head_dim', None)}")
print(f"  qk_rope_head_dim={getattr(cfg, 'qk_rope_head_dim', None)}")
print(f"  v_head_dim={getattr(cfg, 'v_head_dim', None)}")

# Hook into all attention modules and report KV shapes
class CaptureHook:
    def __init__(self):
        self.captured = []

    def __call__(self, module, args, kwargs, output):
        # Inspect args + kwargs for K/V tensors
        for i, a in enumerate(args):
            if torch.is_tensor(a):
                self.captured.append((f"arg{i}", tuple(a.shape), str(a.dtype)))
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                self.captured.append((f"kwarg-{k}", tuple(v.shape), str(v.dtype)))

# Find the attention module class
print("\nSearching for attention layers ...")
found = False
for name, mod in model.named_modules():
    if "self_attn" in name and "." in name and name.endswith("self_attn"):
        print(f"  {name}: {type(mod).__name__}")
        # Print parameter shapes
        for p_name, p in mod.named_parameters():
            print(f"    {p_name}: {tuple(p.shape)}")
        if not found:
            # Hook the first layer's attention
            hook = CaptureHook()
            mod.register_forward_hook(hook, with_kwargs=True)
            found = True
            print(f"  -> Hooked {name} for capture")
            break

# Forward pass
print("\nRunning forward pass on 'Hello world'...")
inputs = tok("Hello world this is a test of the DeepSeek model", return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model(**inputs, use_cache=True)

print(f"\nOutput logits shape: {out.logits.shape}")
print(f"Past key values type: {type(out.past_key_values)}")
if out.past_key_values is not None:
    pkv = out.past_key_values
    if hasattr(pkv, "key_cache"):
        print(f"  pkv.key_cache type: {type(pkv.key_cache)}, len={len(pkv.key_cache) if hasattr(pkv.key_cache, '__len__') else '?'}")
        if len(pkv.key_cache) > 0:
            k0 = pkv.key_cache[0]
            print(f"  key_cache[0] shape: {tuple(k0.shape) if torch.is_tensor(k0) else type(k0)}")
        if hasattr(pkv, "value_cache"):
            v0 = pkv.value_cache[0]
            print(f"  value_cache[0] shape: {tuple(v0.shape) if torch.is_tensor(v0) else type(v0)}")
    elif isinstance(pkv, tuple) and len(pkv) > 0:
        layer0 = pkv[0]
        print(f"  layer0 type: {type(layer0)}")
        if isinstance(layer0, tuple):
            for i, t in enumerate(layer0):
                if torch.is_tensor(t):
                    print(f"    layer0[{i}]: shape {tuple(t.shape)}, dtype {t.dtype}")
                else:
                    print(f"    layer0[{i}]: {type(t)}")

print(f"\nCaptured {len(hook.captured)} attention-call tensors")
for nm, sh, dt in hook.captured[:20]:
    print(f"  {nm}: {sh} {dt}")
