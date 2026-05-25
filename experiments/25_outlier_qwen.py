"""Outlier-aware HQMQ on Qwen2.5-7B.

The diagnostic in experiment 23 showed Qwen2.5 has a few extreme-magnitude
outlier chunks per layer (e.g., layer 27 K_max = 647.6 vs median 2.3).
Linear radius quant zeros out the bulk; log-radius hurts the outliers.

Fix: extract outlier chunks at fp16, quantize the rest with HQMQ. Detect
outliers dynamically per-batch via per-head top-X% quantile — no calibration.

Sweep over outlier fractions {0.5%, 1%, 2%, 5%} × HQMQ configs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model_utils import load_model
from src.quantizers.hqmq import HQMQQuantizer
from src.quantizers.outlier_aware import OutlierAwareHQMQQuantizer
from src.quantizers.naive_int4 import NaivePerTokenIntQuantizer
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

    # Baseline: HQMQ at the best Qwen config without outlier handling
    print(f"\n=== Without outlier extraction (baseline) ===")
    for s, r_b in [(24, 3), (24, 4), (96, 4), (192, 6)]:
        q = HQMQQuantizer(n_layers, n_heads, secondary_size=s, radius_bits=r_b, device=device)
        r = wikitext_perplexity(model, tokenizer, q, seq_len=args.seq_len, max_windows=args.max_windows)
        print(f"  hqmq_s{s}_r{r_b}: ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits)")
        results.append({"name": f"hqmq_s{s}_r{r_b}_no_outlier", **r})

    # With outlier extraction at various fractions
    print(f"\n=== With outlier extraction ===")
    for outlier_pct in [0.5, 1.0, 2.0, 5.0]:
        outlier_frac = outlier_pct / 100.0
        for s, r_b in [(24, 3), (24, 4), (96, 4), (192, 6)]:
            inner = HQMQQuantizer(n_layers, n_heads, secondary_size=s, radius_bits=r_b, device=device)
            q = OutlierAwareHQMQQuantizer(inner, outlier_fraction=outlier_frac)
            r = wikitext_perplexity(model, tokenizer, q, seq_len=args.seq_len, max_windows=args.max_windows)
            print(f"  Outlier{outlier_pct:g}%_hqmq_s{s}_r{r_b}: ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits)")
            results.append({"name": f"Out{outlier_pct:g}p_s{s}_r{r_b}", **r})

    print("\n=== Summary ===")
    print(f"{'config':<32s} {'bits':>6s} {'ppl':>10s} {'Δ vs fp16':>11s}")
    sorted_r = sorted(results, key=lambda x: (x["bits_per_value"], x["perplexity"]))
    for r in sorted_r:
        print(f"{r['name']:<32s} {r['bits_per_value']:>6.2f} {r['perplexity']:>10.3f} {r['perplexity']-fp16_ppl:>+11.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"25_outlier_{args.model.split('/')[-1].replace('-','_')}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
