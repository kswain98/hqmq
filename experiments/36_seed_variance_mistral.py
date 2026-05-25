"""Seed-variance ablation for HQMQ on Mistral-7B (replaces Pythia version).

Runs HQMQ at 5 random seeds across 4 configs at 20w × 2048 on WikiText-103,
reports range and std of end-task perplexity per config.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_utils import load_model
from src.quantizers.hqmq import HQMQQuantizer
from src.eval.perplexity import wikitext_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-windows", type=int, default=20)
    args = p.parse_args()

    print(f"Loading {args.model} ...", flush=True)
    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    d_head = cfg.hidden_size // cfg.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv_heads={n_heads}, d_head={d_head}", flush=True)
    device = "cuda"

    seeds = [0, 1, 7, 42, 1337]
    configs = [
        ("hqmq_s24_r3",  24, 3),
        ("hqmq_s96_r4",  96, 4),
        ("hqmq_s192_r4", 192, 4),
        ("hqmq_s192_r6", 192, 6),
    ]

    results = []
    for tag, s, rb in configs:
        ppls = []
        for seed in seeds:
            q = HQMQQuantizer(n_layers=n_layers, n_heads=n_heads,
                              secondary_size=s, radius_bits=rb,
                              device=device, init="random", seed=seed)
            t0 = time.time()
            r = wikitext_perplexity(model, tokenizer, q,
                                    seq_len=args.seq_len, max_windows=args.max_windows)
            ppls.append(r["perplexity"])
            print(f"  {tag} seed={seed}: ppl={r['perplexity']:.3f} ({r['bits_per_value']:.2f} bits, {time.time()-t0:.1f}s)",
                  flush=True)
            del q
            torch.cuda.empty_cache()
        ppl_min = min(ppls)
        ppl_max = max(ppls)
        ppl_mean = sum(ppls) / len(ppls)
        ppl_std = (sum((p - ppl_mean) ** 2 for p in ppls) / len(ppls)) ** 0.5
        cov = ppl_std / ppl_mean
        results.append({
            "config": tag, "bits": r["bits_per_value"], "seeds": seeds,
            "ppls": ppls, "range": [ppl_min, ppl_max],
            "mean": ppl_mean, "std": ppl_std, "cov_pct": cov * 100,
        })
        print(f"  >> {tag}: range {ppl_min:.3f}--{ppl_max:.3f}, mean={ppl_mean:.3f}, std={ppl_std:.3f}, CoV={cov*100:.2f}%",
              flush=True)

    print("\n=== Summary ===", flush=True)
    for r in results:
        print(f"{r['config']:<14s} bits={r['bits']:.2f}  range {r['range'][0]:.3f}--{r['range'][1]:.3f}"
              f"  std={r['std']:.3f}  CoV={r['cov_pct']:.2f}%", flush=True)

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"36_seed_variance_mistral_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
