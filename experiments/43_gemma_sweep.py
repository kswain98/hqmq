"""HQMQ + Med3x sweep on Google's Gemma-2-9B.

Gemma-2-9B (unsloth mirror): 42 layers, 8 KV heads (GQA), d_head=256
(largest d_head in our model lineup; tests HQMQ scaling).
Uses alternating local/global attention — a Google-specific architectural choice.
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
from src.eval.perplexity import wikitext_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/gemma-2-9b")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-windows", type=int, default=20)
    args = p.parse_args()

    print(f"Loading {args.model} ...", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    d_head = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv_heads={n_heads}, d_head={d_head}", flush=True)
    print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
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
            print(f"  FAILED: {str(e)[:300]}", flush=True)
            results.append({"name": name, "error": str(e)[:500]})
            return None

    print("\n=== fp16 ===", flush=True)
    fp16_ppl = run("fp16", None)

    for bits in [4, 3]:
        print(f"\n=== int{bits} ===", flush=True)
        run(f"int{bits}", NaivePerTokenIntQuantizer(bits=bits))

    # HQMQ Pareto sweep
    for s, rb in [(24, 3), (48, 4), (96, 4), (192, 4)]:
        name = f"hqmq_s{s}_r{rb}"
        print(f"\n=== {name} ===", flush=True)
        q = HQMQQuantizer(n_layers=n_layers, n_heads=n_heads,
                          secondary_size=s, radius_bits=rb, device=device, init="random")
        run(name, q)
        del q
        torch.cuda.empty_cache()

    # Med3x variants
    for s in [24, 96]:
        name = f"hqmq_s{s}_r6_Med3"
        print(f"\n=== {name} ===", flush=True)
        inner = HQMQQuantizer(n_layers=n_layers, n_heads=n_heads,
                              secondary_size=s, radius_bits=6, device=device, init="random")
        run(name, OutlierAwareHQMQQuantizer(inner, threshold_mode="median_mult", median_mult=3.0))
        torch.cuda.empty_cache()

    print("\n=== Summary (sorted by bits) ===", flush=True)
    for r in sorted([r for r in results if "perplexity" in r], key=lambda x: x["bits_per_value"]):
        delta = r["perplexity"] - (fp16_ppl or 0)
        print(f"{r['name']:<24s} {r['bits_per_value']:>6.2f} {r['perplexity']:>10.3f} {delta:>+10.3f}",
              flush=True)

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"43_gemma_sweep_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
