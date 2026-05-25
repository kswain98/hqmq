"""Downstream task scores: PIQA, HellaSwag, ARC-Easy with HQMQ at varying bits.

Validates that the perplexity gains translate to actual downstream quality,
the metric ICLR reviewers will care about most.
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
from src.eval.downstream import eval_piqa, eval_hellaswag, eval_arc_easy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--max-examples", type=int, default=500)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--tasks", nargs="+", default=["piqa", "hellaswag", "arc_easy"])
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
        ("hqmq_s96_r4", HQMQQuantizer(n_layers, n_heads, secondary_size=96, radius_bits=4, device=device)),
        ("hqmq_s192_r4", HQMQQuantizer(n_layers, n_heads, secondary_size=192, radius_bits=4, device=device)),
    ]

    task_fns = {"piqa": eval_piqa, "hellaswag": eval_hellaswag, "arc_easy": eval_arc_easy}
    all_results = []
    for name, q in configs:
        bits = q.bits_per_value() if q is not None else 16.0
        print(f"\n=== {name} ({bits:.2f} bits) ===")
        cfg_results = {"name": name, "bits": bits, "tasks": {}}
        for task in args.tasks:
            t0 = time.time()
            try:
                r = task_fns[task](model, tokenizer, q, max_examples=args.max_examples, device=device)
                cfg_results["tasks"][task] = r
                print(f"  {task}: acc = {r['acc']:.3f}  (n={r['n']}, {time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  {task}: FAILED — {e}")
                cfg_results["tasks"][task] = {"error": str(e)}
        all_results.append(cfg_results)

    print("\n=== Summary ===")
    header = f"{'config':<18s} {'bits':>6s}"
    for task in args.tasks:
        header += f" {task:>12s}"
    print(header)
    for r in all_results:
        line = f"{r['name']:<18s} {r['bits']:>6.2f}"
        for task in args.tasks:
            acc = r["tasks"].get(task, {}).get("acc", float("nan"))
            line += f" {acc:>12.3f}"
        print(line)

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1].replace("-", "_")
    out_path = out_dir / f"16_downstream_{model_tag}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": all_results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
