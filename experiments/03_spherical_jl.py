"""Third experiment: spherical-codebook + JL residual correction.

Goal: match or approach TurboQuant 3-bit lossless quality.
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
from src.quantizers.spherical import SphericalProductQuantizer, SphericalProductQuantizerJL
from src.eval.perplexity import wikitext_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-1b-deduped")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-windows", type=int, default=20)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print(f"Loading {args.model} ({args.dtype}) ...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=dtype)
    print(f"  loaded in {time.time() - t0:.1f}s")

    configs = [
        ("fp16", None),
        ("int4", NaivePerTokenIntQuantizer(bits=4)),
        ("int3", NaivePerTokenIntQuantizer(bits=3)),
        # spherical without JL (baseline for our method)
        ("sph_r4",        SphericalProductQuantizer(chunk_dim=4, radius_bits=4)),
        ("sph_r2",        SphericalProductQuantizer(chunk_dim=4, radius_bits=2)),
        # spherical + JL residual
        ("sph_r4_jl4",    SphericalProductQuantizerJL(chunk_dim=4, radius_bits=4, jl_dim=4)),
        ("sph_r4_jl2",    SphericalProductQuantizerJL(chunk_dim=4, radius_bits=4, jl_dim=2)),
        ("sph_r2_jl4",    SphericalProductQuantizerJL(chunk_dim=4, radius_bits=2, jl_dim=4)),
        ("sph_r2_jl2",    SphericalProductQuantizerJL(chunk_dim=4, radius_bits=2, jl_dim=2)),
        ("sph_r6_jl4",    SphericalProductQuantizerJL(chunk_dim=4, radius_bits=6, jl_dim=4)),
        ("sph_r8_jl4",    SphericalProductQuantizerJL(chunk_dim=4, radius_bits=8, jl_dim=4)),
    ]

    results = []
    for name, quantizer in configs:
        print(f"\n=== {name} ===")
        t0 = time.time()
        res = wikitext_perplexity(
            model, tokenizer, quantizer=quantizer,
            seq_len=args.seq_len, max_windows=args.max_windows,
        )
        res["wall_time_s"] = time.time() - t0
        print(f"  ppl = {res['perplexity']:.3f}  ({res['n_windows']} windows, "
              f"{res['bits_per_value']:.2f} bits/val, {res['wall_time_s']:.1f}s)")
        results.append({"name": name, **res})

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"03_spherical_jl_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}")

    print("\n=== Summary (sorted by bits) ===")
    fp16_ppl = next(r["perplexity"] for r in results if r["name"] == "fp16")
    sorted_results = sorted(results, key=lambda r: r["bits_per_value"])
    print(f"{'config':<14s} {'bits':>6s} {'ppl':>9s} {'Δ vs fp16':>11s}")
    for r in sorted_results:
        delta = r["perplexity"] - fp16_ppl
        print(f"{r['name']:<14s} {r['bits_per_value']:>6.2f} {r['perplexity']:>9.3f} {delta:>+11.3f}")


if __name__ == "__main__":
    main()
