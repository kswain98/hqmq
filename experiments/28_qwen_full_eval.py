"""Full evaluation suite on Qwen2.5-7B with HQMQ + outlier extraction.

Runs downstream tasks (PIQA, HellaSwag, ARC-Easy), needle-in-haystack at 4k,
and long-context perplexity at 4k. With the best HQMQ+Med3× outlier configs.
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
from src.quantizers.outlier_aware import OutlierAwareHQMQQuantizer
from src.eval.perplexity import wikitext_perplexity
from src.eval.long_context import long_context_perplexity
from src.eval.downstream import eval_piqa, eval_hellaswag, eval_arc_easy
from src.eval.needle import needle_in_haystack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--downstream-max-examples", type=int, default=200)
    p.add_argument("--needle-trials", type=int, default=4)
    p.add_argument("--needle-seq-len", type=int, default=4096)
    p.add_argument("--longctx-seq-len", type=int, default=4096)
    p.add_argument("--longctx-windows", type=int, default=3)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = "cuda"
    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(args.model, dtype=dtype)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    d_head = cfg.hidden_size // cfg.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv_heads={n_heads}, d_head={d_head}")

    configs = [
        ("fp16", None),
        ("int4", NaivePerTokenIntQuantizer(bits=4)),
        ("hqmq_s24_r6_Med3", OutlierAwareHQMQQuantizer(
            HQMQQuantizer(n_layers, n_heads, secondary_size=24, radius_bits=6, device=device),
            threshold_mode="median_mult", median_mult=3.0)),
        ("hqmq_s96_r6_Med3", OutlierAwareHQMQQuantizer(
            HQMQQuantizer(n_layers, n_heads, secondary_size=96, radius_bits=6, device=device),
            threshold_mode="median_mult", median_mult=3.0)),
        ("hqmq_s192_r6_Med3", OutlierAwareHQMQQuantizer(
            HQMQQuantizer(n_layers, n_heads, secondary_size=192, radius_bits=6, device=device),
            threshold_mode="median_mult", median_mult=3.0)),
    ]

    all_results = []
    for name, q in configs:
        bits = q.bits_per_value() if q is not None else 16.0
        print(f"\n========== {name} ({bits:.2f} bits) ==========")
        cfg_results = {"name": name, "bits_initial": bits, "tasks": {}}

        # 1) Downstream tasks
        print(f"\n  --- Downstream tasks ---")
        for task_name, task_fn in [("piqa", eval_piqa), ("hellaswag", eval_hellaswag), ("arc_easy", eval_arc_easy)]:
            t0 = time.time()
            try:
                r = task_fn(model, tokenizer, q, max_examples=args.downstream_max_examples, device=device)
                cfg_results["tasks"][task_name] = r
                print(f"  {task_name}: acc = {r['acc']:.3f}  (n={r['n']}, {time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  {task_name}: FAILED — {str(e)[:200]}")
                cfg_results["tasks"][task_name] = {"error": str(e)}

        # 2) Long-context perplexity
        print(f"\n  --- Long-context perplexity ({args.longctx_seq_len} tokens) ---")
        t0 = time.time()
        try:
            r = long_context_perplexity(
                model, tokenizer, q,
                seq_len=args.longctx_seq_len, max_windows=args.longctx_windows, device=device,
            )
            cfg_results["tasks"]["long_ctx"] = r
            print(f"  long_ctx: ppl = {r['perplexity']:.3f}  (n={r['n_windows']}, {time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  long_ctx: FAILED — {str(e)[:200]}")
            cfg_results["tasks"]["long_ctx"] = {"error": str(e)}

        # 3) Needle in Haystack
        print(f"\n  --- Needle in Haystack ({args.needle_seq_len} tokens) ---")
        t0 = time.time()
        try:
            r = needle_in_haystack(
                model, tokenizer, q,
                seq_len=args.needle_seq_len, depths=[0.25, 0.5, 0.75],
                n_trials_per_depth=args.needle_trials, device=device,
            )
            cfg_results["tasks"]["needle"] = r
            print(f"  needle: acc = {r['acc']:.3f}  (n={r['n']}, {time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  needle: FAILED — {str(e)[:200]}")
            cfg_results["tasks"]["needle"] = {"error": str(e)}

        # Update bits with observed outlier rate
        if hasattr(q, "bits_per_value"):
            cfg_results["bits_observed"] = q.bits_per_value()

        all_results.append(cfg_results)

    print("\n=== Final Summary ===")
    cols = ["piqa", "hellaswag", "arc_easy", "long_ctx", "needle"]
    header = f"{'config':<20s} {'bits':>6s}"
    for c in cols:
        header += f" {c:>10s}"
    print(header)
    for r in all_results:
        bits = r.get("bits_observed", r.get("bits_initial", 16.0))
        line = f"{r['name']:<20s} {bits:>6.2f}"
        for c in cols:
            v = r["tasks"].get(c, {})
            if "acc" in v:
                line += f" {v['acc']:>10.3f}"
            elif "perplexity" in v:
                line += f" {v['perplexity']:>10.3f}"
            else:
                line += f" {'—':>10s}"
        print(line)

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"28_qwen_full_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": all_results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
