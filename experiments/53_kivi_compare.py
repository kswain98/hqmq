"""Head-to-head comparison against KIVI (Liu et al., 2024) on lm-eval tasks.

KIVI evaluates on coqa / truthfulqa_gen / gsm8k via lm-evaluation-harness. We
hook our QuantizedCache into lm-eval's HFLM wrapper so we can run the *same*
eval harness on Mistral-7B and produce comparable numbers.

KIVI's published Table 3 (their paper, arXiv:2402.02750):

    Model        | Precision | CoQA  | TruthfulQA | GSM8K
    Llama-2-7B   | fp16      | 63.88 | 30.76      | 13.50
    Llama-2-7B   | KIVI-2    | 63.05 | 33.95      | 12.74
    Mistral-7B   | fp16      | 67.40 | 30.45      | 38.36
    Mistral-7B   | KIVI-2    | 66.35 | 32.17      | 36.01

We add HQMQ at matched bit budgets (s24_r2 ~= 2.79 bits, closest to KIVI-2's
~2.5; s96_r4 = 3.79 bits, comparable to KIVI-4) on Mistral-7B. Output is a
JSON of per-task scores plus a printed comparison table.

Usage:
    python experiments/53_kivi_compare.py \\
        --model mistralai/Mistral-7B-v0.1 \\
        --tasks coqa,truthfulqa_gen,gsm8k \\
        --limit 200 \\
        --configs fp16,int2,hqmq_s24_r2,hqmq_s96_r4

Note: GSM8K is generative (model.generate); CoQA and TruthfulQA_gen are also
generative. These are slower than the MC-style tasks in 16_downstream.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_utils import load_model, QuantizedCache
from src.quantizers.base import KVQuantizer
from src.quantizers.naive_int4 import NaivePerTokenIntQuantizer
from src.quantizers.hqmq import HQMQQuantizer
from src.quantizers.outlier_aware import OutlierAwareHQMQQuantizer


# ---------------------------------------------------------------------------
# Hooking QuantizedCache into lm-evaluation-harness
# ---------------------------------------------------------------------------

def make_quantized_hflm(model, tokenizer, quantizer):
    """Build an lm-eval HFLM whose forward+generate always uses QuantizedCache.

    We monkey-patch the underlying model's `forward` so any internal call that
    builds a cache (including those launched from generate) sees a
    QuantizedCache. Less invasive than subclassing HFLM and overriding several
    internal hooks, and keeps the eval-harness logic completely untouched.
    """
    from lm_eval.models.huggingface import HFLM

    original_forward = model.forward

    def quantized_forward(*args, **kwargs):
        # If the caller already passed a past_key_values, respect it
        # (lm-eval sets one up for caching across requests). Otherwise inject ours.
        if quantizer is not None:
            pkv = kwargs.get("past_key_values", None)
            if pkv is None or not isinstance(pkv, QuantizedCache):
                kwargs["past_key_values"] = QuantizedCache(quantizer)
                kwargs.setdefault("use_cache", True)
        return original_forward(*args, **kwargs)

    if quantizer is not None:
        model.forward = quantized_forward

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
    # Stash the original forward so we can restore between configs.
    lm._hqmq_original_forward = original_forward
    return lm


def restore_forward(model, lm):
    """Undo the monkey-patch so a clean model is left for the next config."""
    if hasattr(lm, "_hqmq_original_forward"):
        model.forward = lm._hqmq_original_forward


# ---------------------------------------------------------------------------
# Quantizer configs we test
# ---------------------------------------------------------------------------

def build_quantizers(model, device="cuda"):
    """Construct the quantizer configs to compare against KIVI."""
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = (getattr(cfg, "num_key_value_heads", None)
               or cfg.num_attention_heads)
    return {
        # Reference: no quantization
        "fp16":           None,
        # Naive 2-bit per-token int (the obvious dumb baseline)
        "int2":           NaivePerTokenIntQuantizer(bits=2),
        # HQMQ at ~KIVI-2 bit budget (2.79 bits)
        "hqmq_s24_r2":    HQMQQuantizer(n_layers, n_heads, secondary_size=24,
                                        radius_bits=2, device=device, init="random"),
        # HQMQ at ~KIVI-4 bit budget (3.79 bits) — the headline lossless config on Mistral
        "hqmq_s96_r4":    HQMQQuantizer(n_layers, n_heads, secondary_size=96,
                                        radius_bits=4, device=device, init="random"),
        # And HQMQ + Med3 — strictly should be similar on Mistral (no outliers),
        # included to verify the safety-net cost is small.
        "hqmq_s96_r4_Med3": OutlierAwareHQMQQuantizer(
            HQMQQuantizer(n_layers, n_heads, secondary_size=96, radius_bits=4,
                          device=device, init="random"),
            threshold_mode="median_mult", median_mult=3.0,
        ),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(model_name, tasks, limit, configs_filter, out_path):
    print(f"Loading {model_name} (bf16) ...", flush=True)
    model, tokenizer = load_model(model_name, dtype=torch.bfloat16)
    cfg = model.config
    print(f"  n_layers={cfg.num_hidden_layers}, "
          f"n_kv_heads={getattr(cfg, 'num_key_value_heads', None) or cfg.num_attention_heads}, "
          f"d_head={cfg.hidden_size // cfg.num_attention_heads}", flush=True)

    from lm_eval import simple_evaluate

    quantizers = build_quantizers(model)
    if configs_filter:
        wanted = set(configs_filter)
        quantizers = {k: v for k, v in quantizers.items() if k in wanted}

    results = {}
    for name, q in quantizers.items():
        if hasattr(q, "bits_per_value"):
            bits = q.bits_per_value()
        else:
            bits = 16.0 if q is None else 2.0  # int2 baseline
        print(f"\n========== {name} ({bits:.2f} bits) ==========", flush=True)

        # Build a fresh HFLM wrapper with the right monkey-patched forward
        lm = make_quantized_hflm(model, tokenizer, q)

        t0 = time.time()
        out = simple_evaluate(
            model=lm,
            tasks=tasks,
            limit=limit,
            log_samples=False,
            cache_requests=False,
        )
        elapsed = time.time() - t0

        # Strip out the heavy parts of the output, keep just per-task scores
        scores = {}
        for task_name, task_res in out.get("results", {}).items():
            scores[task_name] = {
                k: v for k, v in task_res.items()
                if isinstance(v, (int, float))
            }
        results[name] = {
            "bits_per_value": bits,
            "elapsed_s": elapsed,
            "scores": scores,
        }
        print(f"  scores: {scores}  ({elapsed:.1f}s)", flush=True)

        # Restore clean forward before constructing the next quantizer
        restore_forward(model, lm)
        torch.cuda.empty_cache()

    # Summary table
    print("\n========== KIVI comparison summary ==========", flush=True)
    cols = sorted({m for r in results.values() for m in r["scores"]})
    header = f"{'config':<22s} {'bits':>5s} " + " ".join(f"{c:>14s}" for c in cols)
    print(header, flush=True)
    for name, r in results.items():
        row = f"{name:<22s} {r['bits_per_value']:>5.2f} "
        for c in cols:
            sc = r["scores"].get(c, {})
            # Take the main metric for each task (first numeric value)
            v = next((x for x in sc.values() if isinstance(x, float)), None)
            row += f" {v:>13.4f}" if v is not None else " " * 14
        print(row, flush=True)

    # Save JSON
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": model_name,
        "tasks": tasks,
        "limit": limit,
        "results": results,
        "kivi_published_mistral_7B": {
            "fp16":   {"coqa": 67.40, "truthfulqa_gen": 30.45, "gsm8k": 38.36},
            "KIVI-2": {"coqa": 66.35, "truthfulqa_gen": 32.17, "gsm8k": 36.01},
        },
    }, indent=2))
    print(f"\nSaved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--tasks", default="coqa,truthfulqa_gen,gsm8k",
                   help="Comma-separated lm-eval task names.")
    p.add_argument("--limit", type=int, default=200,
                   help="Max examples per task (None = full dataset).")
    p.add_argument("--configs", default="",
                   help="Comma-separated subset of configs to run "
                        "(default: all). Choices: fp16, int2, hqmq_s24_r2, "
                        "hqmq_s96_r4, hqmq_s96_r4_Med3.")
    p.add_argument("--out", default=None, help="Output JSON path.")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    if args.out is None:
        ts = int(time.time())
        model_tag = args.model.replace("/", "_")
        args.out = f"runs/53_kivi_compare_{model_tag}_{ts}.json"

    run(args.model, tasks, args.limit, configs, args.out)


if __name__ == "__main__":
    main()
