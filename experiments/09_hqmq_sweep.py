"""HQMQ codebook-size × radius-bits sweep.

Maps the bit/quality curve for HQMQ without calibration (random init was already
near-optimal in 08_hqmq.py). Tests s={24, 48, 96, 192} × r={2, 3, 4, 6, 8}.

Configurable per model — Pythia-1B fast, Pythia-2.8B as a scale-up check.
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
from src.quantizers.spherical import SphericalProductQuantizerJL
from src.quantizers.hqmq import HQMQQuantizer
from src.eval.perplexity import wikitext_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-1b-deduped")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-windows", type=int, default=10)
    p.add_argument("--secondary-sizes", type=int, nargs="+", default=[24, 48, 96, 192])
    p.add_argument("--radius-bits", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = "cuda"
    print(f"Loading {args.model} ({args.dtype}) ...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=dtype)
    print(f"  loaded in {time.time() - t0:.1f}s")
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    d_head = cfg.hidden_size // cfg.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv_heads={n_heads}, d_head={d_head}")

    results = []

    # Reference baselines
    print(f"\n=== fp16 ===")
    r = wikitext_perplexity(model, tokenizer, None, seq_len=args.seq_len, max_windows=args.max_windows)
    print(f"  ppl = {r['perplexity']:.3f}")
    results.append({"name": "fp16", **r})
    fp16_ppl = r["perplexity"]

    for bits in [4, 3, 2]:
        name = f"int{bits}"
        print(f"\n=== {name} ===")
        r = wikitext_perplexity(model, tokenizer, NaivePerTokenIntQuantizer(bits=bits),
                                seq_len=args.seq_len, max_windows=args.max_windows)
        print(f"  ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits)")
        results.append({"name": name, **r})

    print(f"\n=== sph_r4_jl4 ===")
    r = wikitext_perplexity(
        model, tokenizer, SphericalProductQuantizerJL(chunk_dim=4, radius_bits=4, jl_dim=4),
        seq_len=args.seq_len, max_windows=args.max_windows,
    )
    print(f"  ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits)")
    results.append({"name": "sph_r4_jl4", **r})

    # HQMQ sweep
    for s_size in args.secondary_sizes:
        for r_bits in args.radius_bits:
            name = f"hqmq_s{s_size}_r{r_bits}"
            print(f"\n=== {name} ===")
            hqmq = HQMQQuantizer(
                n_layers=n_layers, n_heads=n_heads,
                secondary_size=s_size, radius_bits=r_bits,
                device=device, init="random",
            )
            t0 = time.time()
            r = wikitext_perplexity(model, tokenizer, hqmq,
                                    seq_len=args.seq_len, max_windows=args.max_windows)
            elapsed = time.time() - t0
            print(f"  ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits, {elapsed:.1f}s)")
            results.append({"name": name, **r})

    print("\n=== Summary (sorted by bits) ===")
    sorted_r = sorted(results, key=lambda x: x["bits_per_value"])
    print(f"{'config':<22s} {'bits':>6s} {'ppl':>9s} {'Δ vs fp16':>11s}")
    for r in sorted_r:
        delta = r["perplexity"] - fp16_ppl
        print(f"{r['name']:<22s} {r['bits_per_value']:>6.2f} {r['perplexity']:>9.3f} {delta:>+11.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1].replace("-", "_")
    out_path = out_dir / f"09_hqmq_sweep_{model_tag}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
