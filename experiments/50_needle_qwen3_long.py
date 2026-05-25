"""Long-context Needle-in-Haystack on Qwen3-8B (16k and 32k contexts)."""
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
from src.eval.needle import needle_in_haystack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[16384, 32768])
    p.add_argument("--n-trials-per-depth", type=int, default=3)
    p.add_argument("--depths", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    args = p.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    d_head = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv_heads={n_heads}, d_head={d_head}")
    device = "cuda"

    configs = [
        ("fp16", None),
        ("int4", NaivePerTokenIntQuantizer(bits=4)),
        ("hqmq_s96_r6_Med3", OutlierAwareHQMQQuantizer(
            HQMQQuantizer(n_layers, n_heads, secondary_size=96, radius_bits=6,
                          device=device, init="random"),
            threshold_mode="median_mult", median_mult=3.0)),
    ]

    all_results = []
    for seq_len in args.seq_lens:
        print(f"\n========== seq_len = {seq_len} ==========")
        for name, q in configs:
            bits = q.bits_per_value() if q is not None else 16.0
            print(f"\n  {name} ({bits:.2f} bits)")
            t0 = time.time()
            try:
                r = needle_in_haystack(
                    model, tokenizer, q, seq_len=seq_len,
                    depths=args.depths, n_trials_per_depth=args.n_trials_per_depth,
                    device=device,
                )
                elapsed = time.time() - t0
                print(f"    acc = {r['acc']:.3f}  (n={r['n']}, {elapsed:.1f}s)")
                all_results.append({"seq_len": seq_len, "name": name, "bits": bits, **r})
            except Exception as e:
                print(f"    FAILED: {str(e)[:200]}")
                all_results.append({"seq_len": seq_len, "name": name, "bits": bits,
                                    "error": str(e)[:500]})
            torch.cuda.empty_cache()

    print("\n=== Summary ===")
    for r in all_results:
        if "acc" in r:
            print(f"  seq={r['seq_len']:>5}  {r['name']:<24s} bits={r['bits']:>6.2f}  acc={r['acc']:.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"50_needle_qwen3_long_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": all_results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
