"""Calibration-free axis: multi-seed HQMQ + additive-VQ comparison.

Two ablations:
  A) HQMQ at 5 different random seeds for the secondary codebook → measures ppl
     variance, validates "no calibration needed" claim.
  B) Additive-VQ (CommVQ-style structure, no training) vs HQMQ at matched bits
     → shows the multiplicative composition is structurally better than the
     additive composition when neither is trained.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_utils import load_model
from src.quantizers.naive_int4 import NaivePerTokenIntQuantizer
from src.quantizers.hqmq import HQMQQuantizer
from src.quantizers.additive_vq import AdditiveVQQuantizer
from src.eval.perplexity import wikitext_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-1b-deduped")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-windows", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 7, 1337])
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
    fp16_ppl = r["perplexity"]
    results.append({"name": "fp16", **r})

    # Part A: Multi-seed HQMQ robustness
    print(f"\n========== PART A: Multi-seed HQMQ (no calibration) ==========")
    for radius_bits in [3, 4]:
        for s_size in [24, 48]:
            ppls = []
            for seed in args.seeds:
                q = HQMQQuantizer(n_layers, n_heads, secondary_size=s_size, radius_bits=radius_bits,
                                  device=device, seed=seed)
                r = wikitext_perplexity(model, tokenizer, q, seq_len=args.seq_len, max_windows=args.max_windows)
                ppls.append(r["perplexity"])
                print(f"  hqmq_s{s_size}_r{radius_bits}_seed{seed}: ppl={r['perplexity']:.3f} ({r['bits_per_value']:.2f} bits)")
                results.append({"name": f"hqmq_s{s_size}_r{radius_bits}_seed{seed}", **r})
            mean_p, std_p = statistics.mean(ppls), statistics.stdev(ppls) if len(ppls) > 1 else 0.0
            print(f"  → s{s_size}_r{radius_bits}: mean={mean_p:.3f}, std={std_p:.4f} (cov {std_p/mean_p*100:.2f}%)")

    # Part B: HQMQ vs Additive-VQ at matched bits
    print(f"\n========== PART B: HQMQ vs Additive-VQ (matched bits, no training) ==========")
    # Matched setups:
    # HQMQ s24 (joint 576) = log2(576) = 9.17 → matches 2 codebooks of size 24 (2*4.58 = 9.17)
    # HQMQ s48 (joint 1152) ≈ 10.17 → matches 2 codebooks of size 48 OR 1 codebook of size 1152
    pairs = [
        ("hqmq_s24_r4",   HQMQQuantizer(n_layers, n_heads, secondary_size=24, radius_bits=4, device=device)),
        ("addvq_2x24_r4", AdditiveVQQuantizer(n_layers, n_heads, n_codebooks=2, codebook_size=24, radius_bits=4, device=device)),
        ("hqmq_s48_r4",   HQMQQuantizer(n_layers, n_heads, secondary_size=48, radius_bits=4, device=device)),
        ("addvq_2x48_r4", AdditiveVQQuantizer(n_layers, n_heads, n_codebooks=2, codebook_size=48, radius_bits=4, device=device)),
        ("hqmq_s24_r3",   HQMQQuantizer(n_layers, n_heads, secondary_size=24, radius_bits=3, device=device)),
        ("addvq_2x24_r3", AdditiveVQQuantizer(n_layers, n_heads, n_codebooks=2, codebook_size=24, radius_bits=3, device=device)),
    ]
    for name, q in pairs:
        print(f"\n=== {name} ===")
        t0 = time.time()
        r = wikitext_perplexity(model, tokenizer, q, seq_len=args.seq_len, max_windows=args.max_windows)
        print(f"  ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits, {time.time()-t0:.1f}s)")
        results.append({"name": name, **r})

    print("\n=== Summary ===")
    print(f"{'config':<28s} {'bits':>6s} {'ppl':>9s} {'Δ vs fp16':>11s}")
    sorted_r = sorted(results, key=lambda x: (x["bits_per_value"], x["name"]))
    for r in sorted_r:
        print(f"{r['name']:<28s} {r['bits_per_value']:>6.2f} {r['perplexity']:>9.3f} {r['perplexity']-fp16_ppl:>+11.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1].replace("-", "_")
    out_path = out_dir / f"14_calibration_free_{model_tag}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
