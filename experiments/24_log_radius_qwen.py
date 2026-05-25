"""Test log-spaced (geometric) radius quantization on Qwen2.5-7B.

Diagnosis: Qwen2.5 has chunks with ||K|| spanning 1000× dynamic range in
some layers. Linear radius quant zeros out the bulk (median quantizes to 0)
because per-token max is dominated by outliers. Log-spaced quant gives
distinct codes to both bulk and outliers in the same bit budget.
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

    print(f"\n=== Linear radius (baseline, all broken) ===")
    for cfg_tuple in [(24, 3), (24, 4), (24, 6), (96, 4), (96, 6), (192, 4), (192, 6)]:
        s_size, r_bits = cfg_tuple
        q = HQMQQuantizer(n_layers, n_heads, secondary_size=s_size, radius_bits=r_bits, log_radius=False, device=device)
        r = wikitext_perplexity(model, tokenizer, q, seq_len=args.seq_len, max_windows=args.max_windows)
        print(f"  hqmq_s{s_size}_r{r_bits}_LIN: ppl = {r['perplexity']:.3f}")
        results.append({"name": f"hqmq_s{s_size}_r{r_bits}_LIN", **r})

    print(f"\n=== Log-spaced radius (the fix) ===")
    for cfg_tuple in [(24, 3), (24, 4), (24, 6), (48, 4), (96, 4), (96, 6), (192, 4), (192, 6)]:
        s_size, r_bits = cfg_tuple
        q = HQMQQuantizer(n_layers, n_heads, secondary_size=s_size, radius_bits=r_bits, log_radius=True, device=device)
        r = wikitext_perplexity(model, tokenizer, q, seq_len=args.seq_len, max_windows=args.max_windows)
        print(f"  hqmq_s{s_size}_r{r_bits}_LOG: ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits)")
        results.append({"name": f"hqmq_s{s_size}_r{r_bits}_LOG", **r})

    # Also: log-radius + Hadamard, just to see if they stack
    print(f"\n=== Log-radius + Hadamard (do they stack?) ===")
    for s_size, r_bits in [(24, 4), (96, 4), (192, 6)]:
        inner = HQMQQuantizer(n_layers, n_heads, secondary_size=s_size, radius_bits=r_bits, log_radius=True, device=device)
        q = HadamardKVQuantizer(inner, d_head=d_head, device=device)
        r = wikitext_perplexity(model, tokenizer, q, seq_len=args.seq_len, max_windows=args.max_windows)
        print(f"  Had_hqmq_s{s_size}_r{r_bits}_LOG: ppl = {r['perplexity']:.3f}")
        results.append({"name": f"Had_hqmq_s{s_size}_r{r_bits}_LOG", **r})

    print("\n=== Summary ===")
    print(f"{'config':<32s} {'bits':>6s} {'ppl':>9s} {'Δ vs fp16':>11s}")
    sorted_r = sorted(results, key=lambda x: (x["bits_per_value"], x["perplexity"]))
    for r in sorted_r:
        print(f"{r['name']:<32s} {r['bits_per_value']:>6.2f} {r['perplexity']:>9.3f} {r['perplexity']-fp16_ppl:>+11.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"24_log_radius_{args.model.split('/')[-1].replace('-','_')}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
