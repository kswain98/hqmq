"""HQMQ + naive baselines + outlier-extraction sweep on Llama-3-8B.

Uses NousResearch/Meta-Llama-3-8B (ungated mirror of meta-llama/Meta-Llama-3-8B).
Llama-3-8B is GQA (32 layers, 8 KV heads, d_head=128), so it's plausible that it
shows the same outlier-heavy behavior as Qwen2.5-7B and benefits from Med3× extraction.

Paper-grade: 50 windows × 2048 tokens on WikiText-103 by default. Override for speed.
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
from src.quantizers.outlier_aware import OutlierAwareHQMQQuantizer
from src.eval.perplexity import wikitext_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="NousResearch/Meta-Llama-3-8B")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-windows", type=int, default=50)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--skip-outlier", action="store_true",
                   help="If naive int4 works fine, skip outlier-extraction configs.")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = "cuda"
    print(f"Loading {args.model} ({args.dtype}) ...", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=dtype)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    d_head = cfg.hidden_size // cfg.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv_heads={n_heads}, d_head={d_head}", flush=True)

    results = []

    def run(name, q):
        t0 = time.time()
        try:
            r = wikitext_perplexity(model, tokenizer, q, seq_len=args.seq_len,
                                    max_windows=args.max_windows)
            elapsed = time.time() - t0
            print(f"  ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits, {elapsed:.1f}s)",
                  flush=True)
            results.append({"name": name, **r})
            return r["perplexity"]
        except Exception as e:
            print(f"  FAILED: {str(e)[:200]}", flush=True)
            results.append({"name": name, "error": str(e)[:500]})
            return None

    # 1) Reference fp16
    print(f"\n=== fp16 ===", flush=True)
    fp16_ppl = run("fp16", None)

    # 2) Naive int4/3/2 baselines
    for bits in [4, 3, 2]:
        name = f"int{bits}"
        print(f"\n=== {name} ===", flush=True)
        run(name, NaivePerTokenIntQuantizer(bits=bits))

    # 3) Spherical+JL baseline (TurboQuant-style structure)
    print(f"\n=== sph_r4_jl4 ===", flush=True)
    run("sph_r4_jl4", SphericalProductQuantizerJL(chunk_dim=4, radius_bits=4, jl_dim=4))

    # 4) HQMQ sweep (no outlier extraction)
    hqmq_configs = [
        ("s24",  24,  3),  # 3.04 bits
        ("s48",  48,  4),  # 3.54 bits
        ("s96",  96,  4),  # 3.79 bits
        ("s192", 192, 4),  # 4.04 bits
        ("s192", 192, 6),  # 4.54 bits
    ]
    for tag, s_size, r_bits in hqmq_configs:
        name = f"hqmq_{tag}_r{r_bits}"
        print(f"\n=== {name} ===", flush=True)
        q = HQMQQuantizer(n_layers=n_layers, n_heads=n_heads,
                          secondary_size=s_size, radius_bits=r_bits,
                          device=device, init="random")
        run(name, q)

    # 5) HQMQ + Med3× outlier extraction (in case Llama-3 has outliers)
    if not args.skip_outlier:
        outlier_configs = [
            ("s24",  24,  6),  # 4.42 bits
            ("s96",  96,  6),  # 4.91 bits
            ("s192", 192, 6),  # 5.15 bits
        ]
        for tag, s_size, r_bits in outlier_configs:
            name = f"hqmq_{tag}_r{r_bits}_Med3"
            print(f"\n=== {name} ===", flush=True)
            inner = HQMQQuantizer(n_layers=n_layers, n_heads=n_heads,
                                  secondary_size=s_size, radius_bits=r_bits,
                                  device=device, init="random")
            q = OutlierAwareHQMQQuantizer(inner, threshold_mode="median_mult", median_mult=3.0)
            run(name, q)

    # Summary
    print(f"\n=== Summary (sorted by bits) ===", flush=True)
    sorted_r = sorted([r for r in results if "perplexity" in r],
                      key=lambda x: x["bits_per_value"])
    print(f"{'config':<24s} {'bits':>6s} {'ppl':>10s} {'Δ vs fp16':>11s}", flush=True)
    for r in sorted_r:
        delta = r["perplexity"] - (fp16_ppl or 0)
        print(f"{r['name']:<24s} {r['bits_per_value']:>6.2f} {r['perplexity']:>10.3f} {delta:>+11.3f}",
              flush=True)

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1].replace("-", "_").replace(".", "")
    out_path = out_dir / f"29_llama_sweep_{model_tag}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
