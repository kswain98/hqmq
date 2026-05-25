"""Test robust radius scaling on Qwen2.5-7B.

Naive HQMQ broke on Qwen because of extreme per-token chunk-radius variance —
outlier chunks dominate per-token max and zero out the rest. Hadamard rotation
helped partially. This test uses median+3*MAD instead of max for the per-token
radius scale, which is robust to outlier chunks.

Combines with Hadamard for the best chance on Qwen.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model_utils import load_model
from src.quantizers.naive_int4 import NaivePerTokenIntQuantizer
from src.quantizers.hqmq import HQMQQuantizer
from src.quantizers.hadamard import HadamardKVQuantizer
from src.eval.perplexity import wikitext_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-windows", type=int, default=10)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = "cuda"
    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(args.model, dtype=dtype)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    d_head = cfg.hidden_size // cfg.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv_heads={n_heads}, d_head={d_head}")

    results = []
    r = wikitext_perplexity(model, tokenizer, None, seq_len=args.seq_len, max_windows=args.max_windows)
    fp16_ppl = r["perplexity"]
    print(f"fp16 ppl = {fp16_ppl:.3f}")
    results.append({"name": "fp16", **r})

    sweeps = [
        # (name, config dict for HQMQQuantizer, use_hadamard)
        ("hqmq_s192_r6_robust",   dict(secondary_size=192, radius_bits=6, robust_radius_scale=True), False),
        ("hqmq_s96_r6_robust",    dict(secondary_size=96,  radius_bits=6, robust_radius_scale=True), False),
        ("hqmq_s48_r6_robust",    dict(secondary_size=48,  radius_bits=6, robust_radius_scale=True), False),
        ("hqmq_s24_r6_robust",    dict(secondary_size=24,  radius_bits=6, robust_radius_scale=True), False),
        ("hqmq_s192_r4_robust",   dict(secondary_size=192, radius_bits=4, robust_radius_scale=True), False),
        ("hqmq_s96_r4_robust",    dict(secondary_size=96,  radius_bits=4, robust_radius_scale=True), False),
        ("hqmq_s24_r4_robust",    dict(secondary_size=24,  radius_bits=4, robust_radius_scale=True), False),
        ("hqmq_s24_r3_robust",    dict(secondary_size=24,  radius_bits=3, robust_radius_scale=True), False),
        # Combined: robust + Hadamard
        ("Had_hqmq_s192_r6_rob",  dict(secondary_size=192, radius_bits=6, robust_radius_scale=True), True),
        ("Had_hqmq_s96_r4_rob",   dict(secondary_size=96,  radius_bits=4, robust_radius_scale=True), True),
        ("Had_hqmq_s24_r3_rob",   dict(secondary_size=24,  radius_bits=3, robust_radius_scale=True), True),
    ]

    for name, cfg_dict, use_had in sweeps:
        q = HQMQQuantizer(n_layers, n_heads, device=device, **cfg_dict)
        if use_had:
            q = HadamardKVQuantizer(q, d_head=d_head, device=device)
        r = wikitext_perplexity(model, tokenizer, q, seq_len=args.seq_len, max_windows=args.max_windows)
        print(f"{name}: ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits)")
        results.append({"name": name, **r})

    print("\n=== Summary ===")
    print(f"{'config':<28s} {'bits':>6s} {'ppl':>9s} {'Δ vs fp16':>11s}")
    sorted_r = sorted(results, key=lambda x: (x["bits_per_value"], x["perplexity"]))
    for r in sorted_r:
        print(f"{r['name']:<28s} {r['bits_per_value']:>6.2f} {r['perplexity']:>9.3f} {r['perplexity']-fp16_ppl:>+11.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"21_robust_radius_{args.model.split('/')[-1].replace('-', '_')}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
