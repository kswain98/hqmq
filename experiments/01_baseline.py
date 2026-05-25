"""First experiment: fp16 baseline + naive per-token int4/int2 on Pythia-1B.

Establishes the upper and lower bounds we'll position our method against.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_utils import load_model
from src.quantizers.identity import IdentityQuantizer
from src.quantizers.naive_int4 import NaivePerTokenIntQuantizer
from src.eval.perplexity import wikitext_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-1b-deduped")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-windows", type=int, default=20, help="Cap for fast iteration; raise for paper-grade.")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16",
                   help="bf16 default — Pythia is numerically unstable in fp16.")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print(f"Loading {args.model} ({args.dtype}) ...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=dtype)
    print(f"  loaded in {time.time() - t0:.1f}s, on {next(model.parameters()).device}")

    configs = [
        ("fp16", None),
        ("identity", IdentityQuantizer()),
        ("int8", NaivePerTokenIntQuantizer(bits=8)),
        ("int4", NaivePerTokenIntQuantizer(bits=4)),
        ("int3", NaivePerTokenIntQuantizer(bits=3)),
        ("int2", NaivePerTokenIntQuantizer(bits=2)),
    ]

    results = []
    for name, quantizer in configs:
        print(f"\n=== {name} ===")
        t0 = time.time()
        res = wikitext_perplexity(
            model, tokenizer, quantizer=quantizer,
            seq_len=args.seq_len, max_windows=args.max_windows,
        )
        res["wall_time_s"] = time.time() - t0
        print(f"  ppl = {res['perplexity']:.3f}  ({res['n_windows']} windows, "
              f"{res['bits_per_value']:.1f} bits/val, {res['wall_time_s']:.1f}s)")
        results.append({"name": name, **res})

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"01_baseline_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}")

    print("\n=== Summary ===")
    fp16_ppl = next(r["perplexity"] for r in results if r["name"] == "fp16")
    print(f"{'config':<10s} {'bits':>5s} {'ppl':>8s} {'Δ vs fp16':>11s}")
    for r in results:
        delta = r["perplexity"] - fp16_ppl
        print(f"{r['name']:<10s} {r['bits_per_value']:>5.1f} {r['perplexity']:>8.3f} {delta:>+11.3f}")


if __name__ == "__main__":
    main()
