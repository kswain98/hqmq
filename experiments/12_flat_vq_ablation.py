"""HQMQ vs Flat-VQ ablation.

Tests whether the multiplicative quaternion composition gives a *quality* advantage
over a flat (independent random) codebook at matched size. If HQMQ matches Flat-VQ,
the structure is a storage trick (24× fewer params). If HQMQ exceeds Flat-VQ,
quaternion multiplication is doing something quality-positive.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_utils import load_model
from src.quantizers.naive_int4 import NaivePerTokenIntQuantizer
from src.quantizers.hqmq import HQMQQuantizer
from src.quantizers.flat_vq import FlatSphericalQuantizer
from src.eval.perplexity import wikitext_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-1b-deduped")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-windows", type=int, default=20)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = "cuda"
    print(f"Loading {args.model} ({args.dtype}) ...")
    model, tokenizer = load_model(args.model, dtype=dtype)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv_heads={n_heads}")

    results = []

    print(f"\n=== fp16 ===")
    r = wikitext_perplexity(model, tokenizer, None, seq_len=args.seq_len, max_windows=args.max_windows)
    print(f"  ppl = {r['perplexity']:.3f}")
    results.append({"name": "fp16", **r})
    fp16_ppl = r["perplexity"]

    # Matched-size comparisons: HQMQ vs Flat-VQ at codebook_size ∈ {24*S | S=24, 48, 96, 192}
    matched = [(24, 24*24), (48, 24*48), (96, 24*96), (192, 24*192)]
    for radius_bits in [3, 4]:
        for s_size, flat_size in matched:
            # HQMQ
            name_h = f"hqmq_s{s_size}_r{radius_bits}"
            print(f"\n=== {name_h} ===")
            q = HQMQQuantizer(n_layers=n_layers, n_heads=n_heads,
                              secondary_size=s_size, radius_bits=radius_bits, device=device)
            t0 = time.time()
            r = wikitext_perplexity(model, tokenizer, q,
                                    seq_len=args.seq_len, max_windows=args.max_windows)
            print(f"  ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits, {time.time()-t0:.1f}s)")
            results.append({"name": name_h, **r})

            # Flat-VQ
            name_f = f"flat_K{flat_size}_r{radius_bits}"
            print(f"\n=== {name_f} ===")
            q = FlatSphericalQuantizer(n_layers=n_layers, n_heads=n_heads,
                                       codebook_size=flat_size, radius_bits=radius_bits, device=device)
            t0 = time.time()
            r = wikitext_perplexity(model, tokenizer, q,
                                    seq_len=args.seq_len, max_windows=args.max_windows)
            print(f"  ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits, {time.time()-t0:.1f}s)")
            results.append({"name": name_f, **r})

    print("\n=== Summary (pairs) ===")
    print(f"{'config':<22s} {'bits':>6s} {'ppl':>9s} {'Δ vs fp16':>11s}")
    sorted_r = sorted(results, key=lambda x: x["bits_per_value"])
    for r in sorted_r:
        print(f"{r['name']:<22s} {r['bits_per_value']:>6.2f} {r['perplexity']:>9.3f} {r['perplexity']-fp16_ppl:>+11.3f}")

    print("\n=== HQMQ vs Flat-VQ deltas at matched size ===")
    for radius_bits in [3, 4]:
        for s_size, flat_size in matched:
            h = next((r["perplexity"] for r in results if r["name"] == f"hqmq_s{s_size}_r{radius_bits}"), None)
            f = next((r["perplexity"] for r in results if r["name"] == f"flat_K{flat_size}_r{radius_bits}"), None)
            if h is not None and f is not None:
                diff = h - f
                winner = "HQMQ" if diff < 0 else "Flat-VQ" if diff > 0 else "tie"
                print(f"  K={flat_size}, r={radius_bits}:  HQMQ={h:.3f}  Flat={f:.3f}  Δ={diff:+.3f} ({winner} wins)")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1].replace("-", "_")
    out_path = out_dir / f"12_flat_vq_ablation_{model_tag}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
