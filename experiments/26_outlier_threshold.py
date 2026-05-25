"""Outlier extraction with median-multiplier threshold on Qwen2.5-7B.

Diagnosis: fraction-based outlier extraction (top X%) doesn't catch enough in
Qwen's catastrophic layers where many chunks are outliers. Median-multiplier
threshold (extract chunks with norm > median × C) adapts: catches all
outliers in heavy-tail layers, almost none in normal layers.
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

    # Sweep over median-multipliers
    print(f"\n=== Median-multiplier threshold sweep ===")
    for mult in [5.0, 10.0, 20.0, 50.0, 100.0]:
        for s, r_b in [(24, 3), (24, 4), (96, 4), (192, 6)]:
            inner = HQMQQuantizer(n_layers, n_heads, secondary_size=s, radius_bits=r_b, device=device)
            q = OutlierAwareHQMQQuantizer(inner, threshold_mode="median_mult", median_mult=mult)
            r = wikitext_perplexity(model, tokenizer, q, seq_len=args.seq_len, max_windows=args.max_windows)
            print(f"  OutMed{mult:.0f}×_hqmq_s{s}_r{r_b}: ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits, ~{q._observed_outlier_rate*100:.2f}% outliers)")
            results.append({"name": f"OutMed{mult:.0f}_s{s}_r{r_b}", **r,
                            "observed_outlier_rate": q._observed_outlier_rate})

    print("\n=== Summary (best per bit budget) ===")
    sorted_r = sorted(results, key=lambda x: (x["bits_per_value"], x["perplexity"]))
    print(f"{'config':<26s} {'bits':>6s} {'ppl':>10s} {'outl%':>7s} {'Δ vs fp16':>11s}")
    for r in sorted_r:
        out_pct = r.get("observed_outlier_rate", 0.0) * 100
        print(f"{r['name']:<26s} {r['bits_per_value']:>6.2f} {r['perplexity']:>10.3f} {out_pct:>7.2f} {r['perplexity']-fp16_ppl:>+11.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"26_outlier_threshold_{args.model.split('/')[-1].replace('-','_')}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
