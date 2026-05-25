"""Push the outlier-aware HQMQ Pareto on Qwen2.5.

Tries tighter median multipliers (3×, 4×, 5×) which give more outliers and
might recover further toward fp16. Also tests smaller HQMQ configs combined
with outlier extraction to map the Pareto frontier.

Also: verify outlier extraction doesn't HURT on Mistral-7B (where HQMQ already
works without it).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model_utils import load_model
from src.quantizers.hqmq import HQMQQuantizer
from src.quantizers.outlier_aware import OutlierAwareHQMQQuantizer
from src.eval.perplexity import wikitext_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-windows", type=int, default=10)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = "cuda"

    all_results = {}

    for model_name in ["Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1"]:
        print(f"\n========== {model_name} ==========")
        model, tokenizer = load_model(model_name, dtype=dtype)
        cfg = model.config
        n_layers = cfg.num_hidden_layers
        n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        d_head = cfg.hidden_size // cfg.num_attention_heads
        print(f"  n_layers={n_layers}, n_kv_heads={n_heads}, d_head={d_head}")

        results = []
        r = wikitext_perplexity(model, tokenizer, None, seq_len=args.seq_len, max_windows=args.max_windows)
        fp16_ppl = r["perplexity"]
        print(f"fp16 ppl = {fp16_ppl:.3f}")
        results.append({"name": "fp16", **r})

        # Push tighter thresholds + smaller configs
        configs = [
            # (s, r, mult) — None for no outlier extraction
            (192, 6, None),  (192, 6, 3.0), (192, 6, 4.0), (192, 6, 5.0),
            (192, 4, None),  (192, 4, 3.0), (192, 4, 5.0),
            (96, 6, None),   (96, 6, 3.0), (96, 6, 5.0),
            (96, 4, None),   (96, 4, 3.0), (96, 4, 5.0),
            (48, 4, None),   (48, 4, 3.0), (48, 4, 5.0),
            (24, 4, None),   (24, 4, 3.0), (24, 4, 5.0),
            (24, 6, 3.0), (24, 6, 5.0),
        ]
        for s, rb, mult in configs:
            inner = HQMQQuantizer(n_layers, n_heads, secondary_size=s, radius_bits=rb, device=device)
            if mult is None:
                q = inner
                tag = f"s{s}_r{rb}_NoOut"
            else:
                q = OutlierAwareHQMQQuantizer(inner, threshold_mode="median_mult", median_mult=mult)
                tag = f"s{s}_r{rb}_Med{mult:.0f}x"
            r = wikitext_perplexity(model, tokenizer, q, seq_len=args.seq_len, max_windows=args.max_windows)
            out_pct = q._observed_outlier_rate * 100 if hasattr(q, "_observed_outlier_rate") and q._n_observations > 0 else 0
            print(f"  {tag:<22s}: ppl = {r['perplexity']:>9.3f}  ({r['bits_per_value']:.2f} bits, ~{out_pct:.2f}% outliers)")
            results.append({"name": tag, **r, "observed_outlier_rate": out_pct / 100})

        # Print summary sorted by bits
        print(f"\n=== Summary for {model_name} (sorted by bits) ===")
        print(f"{'config':<24s} {'bits':>6s} {'ppl':>10s} {'Δ vs fp16':>11s}")
        sorted_r = sorted(results, key=lambda x: (x["bits_per_value"], x["perplexity"]))
        for r in sorted_r:
            print(f"{r['name']:<24s} {r['bits_per_value']:>6.2f} {r['perplexity']:>10.3f} {r['perplexity']-fp16_ppl:>+11.3f}")

        all_results[model_name] = results
        del model
        torch.cuda.empty_cache()

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"27_outlier_pareto_{int(time.time())}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
