"""5-shot MMLU evaluation on Mistral-7B under HQMQ vs naive int4 vs fp16."""
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
from src.eval.mmlu import eval_mmlu_5shot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--max-examples", type=int, default=500)
    args = p.parse_args()

    print(f"Loading {args.model} ...", flush=True)
    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv_heads={n_heads}", flush=True)
    device = "cuda"

    configs = [
        ("fp16", None),
        ("int4", NaivePerTokenIntQuantizer(bits=4)),
        ("int3", NaivePerTokenIntQuantizer(bits=3)),
        ("hqmq_s96_r4", HQMQQuantizer(n_layers=n_layers, n_heads=n_heads,
                                      secondary_size=96, radius_bits=4,
                                      device=device, init="random")),
        ("hqmq_s192_r6_Med3", OutlierAwareHQMQQuantizer(
            HQMQQuantizer(n_layers=n_layers, n_heads=n_heads,
                          secondary_size=192, radius_bits=6, device=device, init="random"),
            threshold_mode="median_mult", median_mult=3.0)),
    ]

    results = []
    for name, q in configs:
        bits = q.bits_per_value() if q is not None else 16.0
        print(f"\n=== {name} ({bits:.2f} bits) ===", flush=True)
        t0 = time.time()
        try:
            r = eval_mmlu_5shot(model, tokenizer, q, max_examples=args.max_examples, device=device)
            elapsed = time.time() - t0
            print(f"  acc = {r['acc']:.4f}  (n={r['n']}, {elapsed:.1f}s)", flush=True)
            results.append({"name": name, "bits": bits, "acc": r["acc"], "n": r["n"]})
        except Exception as e:
            print(f"  FAILED: {str(e)[:300]}", flush=True)
            results.append({"name": name, "bits": bits, "error": str(e)[:500]})

    print("\n=== Summary ===", flush=True)
    fp16_acc = next((r["acc"] for r in results if r["name"] == "fp16"), None)
    print(f"{'config':<24s} {'bits':>6s} {'acc':>8s} {'Δ vs fp16':>11s}", flush=True)
    for r in results:
        if "acc" in r:
            delta = (r["acc"] - (fp16_acc or 0)) * 100
            print(f"{r['name']:<24s} {r['bits']:>6.2f} {r['acc']:>8.4f} {delta:>+10.2f}%",
                  flush=True)

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"48_mmlu_mistral_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
