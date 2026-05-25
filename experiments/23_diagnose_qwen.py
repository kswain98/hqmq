"""Diagnose why Qwen2.5-7B's KV cache resists quantization.

Compares KV-tensor statistics (per-chunk norm distribution, channel outliers)
between Mistral-7B-v0.1 (HQMQ works) and Qwen2.5-7B (HQMQ fails).

The hope: find a concrete numeric difference (e.g., 100× wider chunk-norm
distribution in Qwen) that explains the failure and points at a fix.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset

from src.model_utils import load_model
from src.eval.kv_stats import probe_kv_stats


def summarize(stats, model_name: str):
    print(f"\n========== {model_name} ==========")
    print(f"{'layer':>5s} {'K_max/med':>11s} {'K_chan_outl':>13s} {'V_max/med':>11s} {'V_chan_outl':>13s} {'K_global_max':>13s} {'K_global_med':>13s}")
    for li in sorted(stats.keys()):
        k = stats[li]["K"]
        v = stats[li]["V"]
        # max/median across all heads for this layer (use max of per-head max-to-med ratio)
        k_ratio = max(k["per_head_max_to_median"])
        v_ratio = max(v["per_head_max_to_median"])
        print(f"{li:>5d} {k_ratio:>11.2f} {k['channel_abs_max_to_mean']:>13.2f} "
              f"{v_ratio:>11.2f} {v['channel_abs_max_to_mean']:>13.2f} "
              f"{k['global_max']:>13.2f} {k['global_median']:>13.2f}")

    print(f"\nAggregate (over all layers):")
    k_ratios = [max(stats[li]["K"]["per_head_max_to_median"]) for li in stats]
    v_ratios = [max(stats[li]["V"]["per_head_max_to_median"]) for li in stats]
    k_chan = [stats[li]["K"]["channel_abs_max_to_mean"] for li in stats]
    print(f"  K max/median ratio: min={min(k_ratios):.2f}, max={max(k_ratios):.2f}, median={sorted(k_ratios)[len(k_ratios)//2]:.2f}")
    print(f"  V max/median ratio: min={min(v_ratios):.2f}, max={max(v_ratios):.2f}, median={sorted(v_ratios)[len(v_ratios)//2]:.2f}")
    print(f"  K channel outlier ratio: min={min(k_chan):.2f}, max={max(k_chan):.2f}, median={sorted(k_chan)[len(k_chan)//2]:.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = "cuda"

    # Get a representative passage
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    sample_text = "\n\n".join(ds["text"][:200])

    all_stats = {}
    for model_name in ["mistralai/Mistral-7B-v0.1", "Qwen/Qwen2.5-7B"]:
        print(f"\nLoading {model_name} ...")
        model, tokenizer = load_model(model_name, dtype=dtype)
        stats = probe_kv_stats(model, tokenizer, sample_text, device=device)
        summarize(stats, model_name)
        all_stats[model_name] = stats
        del model
        torch.cuda.empty_cache()

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "23_kv_diagnosis.json"
    out_path.write_text(json.dumps({m: {str(k): v for k, v in s.items()} for m, s in all_stats.items()}, indent=2))
    print(f"\nSaved {out_path}")

    print("\n=== HEAD-TO-HEAD COMPARISON ===")
    for model_name in ["mistralai/Mistral-7B-v0.1", "Qwen/Qwen2.5-7B"]:
        s = all_stats[model_name]
        k_ratios = [max(s[li]["K"]["per_head_max_to_median"]) for li in s]
        v_ratios = [max(s[li]["V"]["per_head_max_to_median"]) for li in s]
        k_chan = [s[li]["K"]["channel_abs_max_to_mean"] for li in s]
        print(f"{model_name}:")
        print(f"  K max-to-median (chunk norms): mean={sum(k_ratios)/len(k_ratios):.2f}")
        print(f"  V max-to-median (chunk norms): mean={sum(v_ratios)/len(v_ratios):.2f}")
        print(f"  K channel-outlier ratio:       mean={sum(k_chan)/len(k_chan):.2f}")


if __name__ == "__main__":
    main()
