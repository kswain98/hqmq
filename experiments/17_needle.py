"""Needle-in-Haystack retrieval test for HQMQ at varying bits."""

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
from src.eval.needle import needle_in_haystack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[4096])
    p.add_argument("--n-trials-per-depth", type=int, default=4)
    p.add_argument("--depths", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = "cuda"
    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(args.model, dtype=dtype)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv_heads={n_heads}")

    configs = [
        ("fp16", None),
        ("int4", NaivePerTokenIntQuantizer(bits=4)),
        ("int3", NaivePerTokenIntQuantizer(bits=3)),
        ("hqmq_s24_r3", HQMQQuantizer(n_layers, n_heads, secondary_size=24, radius_bits=3, device=device)),
        ("hqmq_s48_r4", HQMQQuantizer(n_layers, n_heads, secondary_size=48, radius_bits=4, device=device)),
        ("hqmq_s192_r4", HQMQQuantizer(n_layers, n_heads, secondary_size=192, radius_bits=4, device=device)),
    ]

    all_results = []
    for seq_len in args.seq_lens:
        print(f"\n========== seq_len = {seq_len} ==========")
        for name, q in configs:
            bits = q.bits_per_value() if q is not None else 16.0
            print(f"\n  === {name} ({bits:.2f} bits) ===")
            t0 = time.time()
            try:
                r = needle_in_haystack(
                    model, tokenizer, q,
                    seq_len=seq_len, depths=args.depths,
                    n_trials_per_depth=args.n_trials_per_depth, device=device,
                )
                print(f"    acc = {r['acc']:.3f}  (n={r['n']}, {time.time()-t0:.1f}s)")
                # Per-depth breakdown
                by_depth = {}
                for tr in r["per_trial"]:
                    by_depth.setdefault(tr["depth"], []).append(tr["correct"])
                for d, hits in sorted(by_depth.items()):
                    print(f"      depth={d:.2f}: {sum(hits)}/{len(hits)} correct")
                all_results.append({"name": name, "bits": bits, "seq_len": seq_len, **r})
            except Exception as e:
                print(f"    FAILED — {e}")
                all_results.append({"name": name, "bits": bits, "seq_len": seq_len, "error": str(e)})

    print("\n=== Summary ===")
    print(f"{'config':<18s} {'bits':>6s} {'seq_len':>8s} {'acc':>6s}")
    for r in all_results:
        acc = r.get("acc", float("nan"))
        print(f"{r['name']:<18s} {r['bits']:>6.2f} {r['seq_len']:>8d} {acc:>6.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1].replace("-", "_")
    out_path = out_dir / f"17_needle_{model_tag}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": all_results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
