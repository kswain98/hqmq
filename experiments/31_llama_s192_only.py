"""Targeted Llama-3-8B follow-up for the s192 HQMQ configs at moderate windows.

The full sweep stalled on s192 configs at 50w × 2048 due to memory pressure
from the large 24*192=4608 codebook gather. Use 20w × 2048 instead.
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
from src.quantizers.outlier_aware import OutlierAwareHQMQQuantizer
from src.eval.perplexity import wikitext_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="NousResearch/Meta-Llama-3-8B")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-windows", type=int, default=20)
    args = p.parse_args()

    print(f"Loading {args.model} (bf16) ...", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    d_head = cfg.hidden_size // cfg.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv_heads={n_heads}, d_head={d_head}", flush=True)

    device = "cuda"
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

    print("\n=== fp16 ===", flush=True)
    fp16_ppl = run("fp16", None)

    configs = [
        ("hqmq_s192_r4", 192, 4, False),
        ("hqmq_s192_r6", 192, 6, False),
        ("hqmq_s24_r6_Med3", 24, 6, True),
        ("hqmq_s96_r6_Med3", 96, 6, True),
        ("hqmq_s192_r6_Med3", 192, 6, True),
    ]
    for name, s, rb, med in configs:
        print(f"\n=== {name} ===", flush=True)
        inner = HQMQQuantizer(n_layers=n_layers, n_heads=n_heads,
                              secondary_size=s, radius_bits=rb, device=device, init="random")
        q = OutlierAwareHQMQQuantizer(inner, threshold_mode="median_mult", median_mult=3.0) if med else inner
        run(name, q)

    print("\n=== Summary ===", flush=True)
    for r in sorted([r for r in results if "perplexity" in r], key=lambda x: x["bits_per_value"]):
        delta = r["perplexity"] - (fp16_ppl or 0)
        print(f"{r['name']:<24s} {r['bits_per_value']:>6.2f} {r['perplexity']:>10.3f} {delta:>+8.3f}",
              flush=True)

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"31_llama_s192_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
