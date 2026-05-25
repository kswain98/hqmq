"""HQMQ with FREE-magnitude secondary codebook — sub-2-bit territory.

When the secondary codebook is not unit-normalized, each codeword carries
direction + magnitude. No per-chunk radius bits needed (only one per-token
fp16 scale, ~0.06 bits/elem at d_head=256).

Bit cost: log2(24*S)/4 per element. For S=24 → 1.15 bits/elem (!).
For S=192 → 2.07 bits/elem.

Targets the 1-2 bit territory where CommVQ currently sits.
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
from src.eval.perplexity import wikitext_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-1b-deduped")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-windows", type=int, default=10)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = "cuda"
    print(f"Loading {args.model} ({args.dtype}) ...")
    model, tokenizer = load_model(args.model, dtype=dtype)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    d_head = cfg.hidden_size // cfg.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv_heads={n_heads}, d_head={d_head}")

    results = []
    print(f"\n=== fp16 ===")
    r = wikitext_perplexity(model, tokenizer, None, seq_len=args.seq_len, max_windows=args.max_windows)
    print(f"  ppl = {r['perplexity']:.3f}")
    results.append({"name": "fp16", **r})
    fp16_ppl = r["perplexity"]

    for bits in [2, 3, 4]:
        r = wikitext_perplexity(model, tokenizer, NaivePerTokenIntQuantizer(bits=bits),
                                seq_len=args.seq_len, max_windows=args.max_windows)
        print(f"int{bits}: ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits)")
        results.append({"name": f"int{bits}", **r})

    # Free-magnitude HQMQ — extreme low-bit territory
    configs = []
    for s_size in [24, 48, 96, 192]:
        configs.append((f"hqmqF_s{s_size}", HQMQQuantizer(
            n_layers=n_layers, n_heads=n_heads,
            secondary_size=s_size, radius_bits=0,  # ignored in free mode
            magnitude_mode="free", device=device,
        )))
        # Also test free + JL residual
        for jl in [2, 4]:
            configs.append((f"hqmqF_s{s_size}_jl{jl}", HQMQQuantizer(
                n_layers=n_layers, n_heads=n_heads,
                secondary_size=s_size, radius_bits=0, jl_dim=jl,
                magnitude_mode="free", device=device,
            )))

    for name, quantizer in configs:
        print(f"\n=== {name} ===")
        t0 = time.time()
        r = wikitext_perplexity(model, tokenizer, quantizer,
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
    out_path = out_dir / f"11_hqmq_free_{model_tag}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
