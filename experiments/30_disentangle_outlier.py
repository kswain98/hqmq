"""Disentangle the contributions of (a) outlier extraction and (b) HQMQ.

Runs on Qwen2.5-7B (where naive int4 collapses to 18,079 ppl):
  - fp16
  - naive int4              (broken)
  - naive int4 + Med3×      (outlier extraction alone)
  - HQMQ s24_r6 (no outlier) (HQMQ alone — also broken on Qwen)
  - HQMQ s24_r6 + Med3×      (both)
  - HQMQ s96_r6 + Med3×, s192_r6 + Med3×

If naive int4 + Med3× is ≪ HQMQ + Med3× then the multiplicative codebook
carries weight; if they are close then the outlier extraction is doing most
of the work and the paper's main claim weakens.
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
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-windows", type=int, default=50)
    args = p.parse_args()

    print(f"Loading {args.model} (bf16) ...", flush=True)
    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
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

    # Naive int4 alone (broken on Qwen)
    print("\n=== int4 ===", flush=True)
    run("int4", NaivePerTokenIntQuantizer(bits=4))

    print("\n=== int3 ===", flush=True)
    run("int3", NaivePerTokenIntQuantizer(bits=3))

    # Naive int4/3 + Med3× outlier extraction
    print("\n=== int4 + Med3× ===", flush=True)
    run("int4_Med3", OutlierAwareGenericQuantizer(
        NaivePerTokenIntQuantizer(bits=4), chunk_dim=4,
        threshold_mode="median_mult", median_mult=3.0))

    print("\n=== int3 + Med3× ===", flush=True)
    run("int3_Med3", OutlierAwareGenericQuantizer(
        NaivePerTokenIntQuantizer(bits=3), chunk_dim=4,
        threshold_mode="median_mult", median_mult=3.0))

    # Tighter outlier filters with naive int
    print("\n=== int4 + Med5× ===", flush=True)
    run("int4_Med5", OutlierAwareGenericQuantizer(
        NaivePerTokenIntQuantizer(bits=4), chunk_dim=4,
        threshold_mode="median_mult", median_mult=5.0))

    # HQMQ no-outlier (broken on Qwen)
    print("\n=== HQMQ s24_r6 (no outlier) ===", flush=True)
    run("hqmq_s24_r6", HQMQQuantizer(
        n_layers=n_layers, n_heads=n_heads,
        secondary_size=24, radius_bits=6, device=device, init="random"))

    # HQMQ + Med3×
    for s in [24, 96, 192]:
        name = f"hqmq_s{s}_r6_Med3"
        print(f"\n=== {name} ===", flush=True)
        run(name, OutlierAwareHQMQQuantizer(
            HQMQQuantizer(n_layers=n_layers, n_heads=n_heads,
                          secondary_size=s, radius_bits=6, device=device, init="random"),
            threshold_mode="median_mult", median_mult=3.0))

    print("\n=== Summary (sorted by bits) ===", flush=True)
    sorted_r = sorted([r for r in results if "perplexity" in r],
                      key=lambda x: x["bits_per_value"])
    print(f"{'config':<28s} {'bits':>6s} {'ppl':>12s} {'Δ vs fp16':>12s}", flush=True)
    for r in sorted_r:
        delta = r["perplexity"] - (fp16_ppl or 0)
        print(f"{r['name']:<28s} {r['bits_per_value']:>6.2f} {r['perplexity']:>12.3f} {delta:>+12.3f}",
              flush=True)

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1].replace("-", "_").replace(".", "")
    out_path = out_dir / f"30_disentangle_outlier_{model_tag}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
