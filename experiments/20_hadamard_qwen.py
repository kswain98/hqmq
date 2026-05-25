"""HQMQ + Hadamard pre-rotation on Qwen2.5-7B.

Naive HQMQ broke on Qwen2.5-7B (and so did naive int4) due to outlier K
channels. This script tests whether a fixed Hadamard pre-rotation rescues
the method, matching VecInfer / QuaRot / SpinQuant's approach.
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

    print(f"\n=== fp16 ===")
    r = wikitext_perplexity(model, tokenizer, None, seq_len=args.seq_len, max_windows=args.max_windows)
    print(f"  ppl = {r['perplexity']:.3f}")
    fp16_ppl = r["perplexity"]
    results.append({"name": "fp16", **r})

    # Reference: naive int4/int3 — these blew up without Hadamard
    for bits in [4, 3]:
        q = NaivePerTokenIntQuantizer(bits=bits)
        r = wikitext_perplexity(model, tokenizer, q, seq_len=args.seq_len, max_windows=args.max_windows)
        print(f"int{bits} (no Hadamard): ppl = {r['perplexity']:.3f}")
        results.append({"name": f"int{bits}_noH", **r})

    # int4/int3 + Hadamard — first sanity that Hadamard fixes naive
    for bits in [4, 3]:
        inner = NaivePerTokenIntQuantizer(bits=bits)
        q = HadamardKVQuantizer(inner, d_head=d_head, device=device)
        r = wikitext_perplexity(model, tokenizer, q, seq_len=args.seq_len, max_windows=args.max_windows)
        print(f"Hadamard+int{bits}: ppl = {r['perplexity']:.3f}")
        results.append({"name": f"Had_int{bits}", **r})

    # HQMQ — first without Hadamard (re-confirm broken), then with Hadamard
    print(f"\n=== HQMQ without Hadamard (re-confirm broken) ===")
    for s_size, r_bits in [(24, 3), (96, 4), (192, 6)]:
        q = HQMQQuantizer(n_layers, n_heads, secondary_size=s_size, radius_bits=r_bits, device=device)
        r = wikitext_perplexity(model, tokenizer, q, seq_len=args.seq_len, max_windows=args.max_windows)
        print(f"hqmq_s{s_size}_r{r_bits}: ppl = {r['perplexity']:.3f}")
        results.append({"name": f"hqmq_s{s_size}_r{r_bits}_noH", **r})

    print(f"\n=== HQMQ with Hadamard ===")
    for s_size, r_bits in [(24, 3), (24, 4), (48, 4), (96, 4), (96, 6), (192, 4), (192, 6)]:
        inner = HQMQQuantizer(n_layers, n_heads, secondary_size=s_size, radius_bits=r_bits, device=device)
        q = HadamardKVQuantizer(inner, d_head=d_head, device=device)
        r = wikitext_perplexity(model, tokenizer, q, seq_len=args.seq_len, max_windows=args.max_windows)
        print(f"Had_hqmq_s{s_size}_r{r_bits}: ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits)")
        results.append({"name": f"Had_hqmq_s{s_size}_r{r_bits}", **r})

    print("\n=== Summary ===")
    print(f"{'config':<28s} {'bits':>6s} {'ppl':>9s} {'Δ vs fp16':>11s}")
    sorted_r = sorted(results, key=lambda x: (x["bits_per_value"], x["perplexity"]))
    for r in sorted_r:
        print(f"{r['name']:<28s} {r['bits_per_value']:>6.2f} {r['perplexity']:>9.3f} {r['perplexity']-fp16_ppl:>+11.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1].replace("-", "_")
    out_path = out_dir / f"20_hadamard_{model_tag}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
