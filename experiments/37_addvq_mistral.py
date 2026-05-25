"""HQMQ vs uncalibrated additive VQ on Mistral-7B (replaces Pythia version).

Additive VQ: c ≈ c1[i1] + c2[i2] with two random codebooks. CommVQ structure
but no training. At matched bits and matched effective codebook size.
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
from src.quantizers.additive_vq import AdditiveVQQuantizer
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
    print(f"  n_layers={n_layers}, n_kv_heads={n_heads}", flush=True)
    device = "cuda"

    results = []

    # Matched-bit comparisons:
    # HQMQ s24 has log2(24*24) = 9.17 direction bits. Additive 2-codebook of size 24
    # has 2*log2(24) = 9.17 direction bits. Both at radius_bits matched.
    # HQMQ s48 ≡ 2*log2(48). HQMQ s192 ≡ 2*log2(192).
    pairs = [
        # (hqmq_secondary_size, additive_codebook_size, radius_bits, target_bits)
        (24,  24,  3, "3.04"),
        (48,  48,  4, "3.54"),
        (192, 192, 4, "4.04"),
    ]

    for s, cb_size, rb, label in pairs:
        # HQMQ
        name_h = f"hqmq_s{s}_r{rb}"
        print(f"\n=== {name_h} (target {label} bits) ===", flush=True)
        q = HQMQQuantizer(n_layers=n_layers, n_heads=n_heads,
                          secondary_size=s, radius_bits=rb, device=device, init="random")
        t0 = time.time()
        r = wikitext_perplexity(model, tokenizer, q,
                                seq_len=args.seq_len, max_windows=args.max_windows)
        print(f"  ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits, {time.time()-t0:.1f}s)",
              flush=True)
        results.append({"name": name_h, **r})
        del q
        torch.cuda.empty_cache()

        # Additive VQ — 2 codebooks of size cb_size each
        name_a = f"add_K{cb_size}x2_r{rb}"
        print(f"\n=== {name_a} (target {label} bits) ===", flush=True)
        q = AdditiveVQQuantizer(n_layers=n_layers, n_heads=n_heads,
                                n_codebooks=2, codebook_size=cb_size,
                                radius_bits=rb, device=device)
        t0 = time.time()
        r = wikitext_perplexity(model, tokenizer, q,
                                seq_len=args.seq_len, max_windows=args.max_windows)
        print(f"  ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits, {time.time()-t0:.1f}s)",
              flush=True)
        results.append({"name": name_a, **r})
        del q
        torch.cuda.empty_cache()

    print("\n=== HQMQ vs Additive-VQ deltas ===", flush=True)
    for s, cb_size, rb, label in pairs:
        h = next((r["perplexity"] for r in results if r["name"] == f"hqmq_s{s}_r{rb}"), None)
        a = next((r["perplexity"] for r in results if r["name"] == f"add_K{cb_size}x2_r{rb}"), None)
        if h is not None and a is not None:
            ratio = a / h
            print(f"  bits={label}:  HQMQ={h:.3f}  Additive={a:.3f}  ratio={ratio:.2f}x", flush=True)

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"37_addvq_mistral_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
