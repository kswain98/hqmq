"""Qwen3-8B with Med3x outlier extraction.

Completes the Qwen3 outlier-rescue story (analog of Qwen2.5 results):
  - HQMQ s24_r6 + Med3x
  - HQMQ s96_r6 + Med3x
  - HQMQ s192_r6 + Med3x  (skip if too memory-heavy)
  - naive int4 + Med3x  (disentanglement: is the multiplicative codebook necessary?)
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
from src.quantizers.outlier_aware import OutlierAwareHQMQQuantizer
from src.quantizers.outlier_generic import OutlierAwareGenericQuantizer
from src.eval.perplexity import wikitext_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-windows", type=int, default=20)
    args = p.parse_args()

    print(f"Loading {args.model} ...", flush=True)
    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    d_head = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
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

    # fp16 reference
    print("\n=== fp16 ===", flush=True)
    fp16_ppl = run("fp16", None)

    # Disentanglement: naive int4 + Med3x (does outlier extraction alone fix Qwen3?)
    print("\n=== int4 + Med3× ===", flush=True)
    run("int4_Med3", OutlierAwareGenericQuantizer(
        NaivePerTokenIntQuantizer(bits=4), chunk_dim=4,
        threshold_mode="median_mult", median_mult=3.0))

    # HQMQ + Med3× at three codebook sizes
    for s in [24, 96]:  # skip 192 to avoid memory issues
        name = f"hqmq_s{s}_r6_Med3"
        print(f"\n=== {name} ===", flush=True)
        run(name, OutlierAwareHQMQQuantizer(
            HQMQQuantizer(n_layers=n_layers, n_heads=n_heads,
                          secondary_size=s, radius_bits=6,
                          device=device, init="random"),
            threshold_mode="median_mult", median_mult=3.0))
        torch.cuda.empty_cache()

    print("\n=== Summary (sorted by bits) ===", flush=True)
    for r in sorted([r for r in results if "perplexity" in r], key=lambda x: x["bits_per_value"]):
        delta = r["perplexity"] - (fp16_ppl or 0)
        print(f"{r['name']:<24s} {r['bits_per_value']:>6.2f} {r['perplexity']:>10.3f} {delta:>+10.3f}",
              flush=True)

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"38_qwen3_med3_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
