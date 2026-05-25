"""RULER long-context evaluation under HQMQ.

RULER (Hsieh et al., 2024) is the standard long-context benchmark for testing
whether a model can actually USE its claimed context window. It has 13 subtasks
across 5 categories; lm-evaluation-harness exposes the headline subset:

    ruler_cwe         common-word extraction
    ruler_fwe         frequent-word extraction
    ruler_qa_hotpot   multi-hop QA across long context
    ruler_qa_squad    extractive QA across long context
    ruler_vt          variable tracking

We run the same QuantizedHFLM wrapper as 53_kivi_compare.py to hook HQMQ
into lm-eval. RULER is sequence-length-parametrized; we test 4k, 8k, and 16k
on Qwen3-8B (the outlier-heavy model where KV quantization matters most).
32k and 64k are flagged in big_gpu_queue.sh as needing >24 GB.

Usage:
    LD_PRELOAD=/home/kswain/miniforge3/envs/tensor/lib/libstdc++.so.6 \\
    python experiments/55_ruler_eval.py \\
        --model Qwen/Qwen3-8B \\
        --seq-lens 4096,8192 \\
        --configs fp16,hqmq_s96_r6_Med3 \\
        --limit 50

The `LD_PRELOAD` is the same libstdc++ shim used for KIVI compare.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_utils import load_model, QuantizedCache
from src.quantizers.naive_int4 import NaivePerTokenIntQuantizer
from src.quantizers.hqmq import HQMQQuantizer
from src.quantizers.outlier_aware import OutlierAwareHQMQQuantizer


def make_quantized_hflm(model, tokenizer, quantizer, max_length):
    """HFLM with QuantizedCache injected via monkey-patched forward.

    Same trick as 53_kivi_compare.py — see that file for the design notes.
    """
    from lm_eval.models.huggingface import HFLM

    original_forward = model.forward

    def quantized_forward(*args, **kwargs):
        if quantizer is not None:
            pkv = kwargs.get("past_key_values", None)
            if pkv is None or not isinstance(pkv, QuantizedCache):
                kwargs["past_key_values"] = QuantizedCache(quantizer)
                kwargs.setdefault("use_cache", True)
        return original_forward(*args, **kwargs)

    if quantizer is not None:
        model.forward = quantized_forward

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1,
              max_length=max_length)
    lm._hqmq_original_forward = original_forward
    return lm


def restore_forward(model, lm):
    if hasattr(lm, "_hqmq_original_forward"):
        model.forward = lm._hqmq_original_forward


def build_quantizers(model, device="cuda"):
    """Subset of configs from 53_kivi_compare.py, focused on the headline
    HQMQ choice for long-context (s96_r6 + Med3x at ~5 bits)."""
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = (getattr(cfg, "num_key_value_heads", None)
               or cfg.num_attention_heads)
    return {
        "fp16":              None,
        "int4":              NaivePerTokenIntQuantizer(bits=4),
        "hqmq_s96_r6_Med3":  OutlierAwareHQMQQuantizer(
            HQMQQuantizer(n_layers, n_heads, secondary_size=96, radius_bits=6,
                          device=device, init="random"),
            threshold_mode="median_mult", median_mult=3.0,
        ),
    }


def run(model_name, tasks, seq_lens, configs_filter, limit, out_path):
    print(f"Loading {model_name} (bf16) ...", flush=True)
    model, tokenizer = load_model(model_name, dtype=torch.bfloat16)
    print(f"  loaded.", flush=True)

    # lm-eval RULER tasks read `max_seq_lengths` AND the tokenizer name from
    # per-task metadata. Without `tokenizer`/`pretrained` in metadata, the
    # qa_utils.get_qa_dataset fallback uses an empty dict which fails inside
    # the @functools.cache-wrapped get_tokenizer (-> "unhashable type: dict").
    # The DEFAULT_SEQ_LENGTHS module-level constant is not honored, so both
    # must come through metadata via a TaskManager.
    print(f"  RULER seq lengths requested: {seq_lens}", flush=True)

    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager
    task_manager = TaskManager(
        metadata={
            "max_seq_lengths": list(seq_lens),
            "tokenizer": model_name,
            "pretrained": model_name,
        },
    )

    quantizers = build_quantizers(model)
    if configs_filter:
        wanted = set(configs_filter)
        quantizers = {k: v for k, v in quantizers.items() if k in wanted}

    results = {}
    for name, q in quantizers.items():
        if hasattr(q, "bits_per_value"):
            bits = q.bits_per_value()
        else:
            bits = 16.0 if q is None else 4.0
        print(f"\n========== {name} ({bits:.2f} bits) ==========", flush=True)

        # Use the longest requested seq_len as the HFLM context cap.
        lm = make_quantized_hflm(model, tokenizer, q,
                                  max_length=max(seq_lens))
        t0 = time.time()
        try:
            out = simple_evaluate(
                model=lm, tasks=tasks,
                limit=limit, log_samples=False, cache_requests=False,
                task_manager=task_manager,
            )
            elapsed = time.time() - t0
            # Each RULER task reports a separate metric per seq_len; collect all.
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
        except Exception as e:
            results[name] = {
                "bits_per_value": bits,
                "error": str(e)[:500],
            }
            print(f"  FAILED: {str(e)[:200]}", flush=True)
        finally:
            restore_forward(model, lm)
            torch.cuda.empty_cache()

    # Summary
    print("\n========== RULER summary ==========", flush=True)
    for name, r in results.items():
        if "error" in r:
            print(f"{name}: ERROR ({r['error'][:80]})", flush=True)
            continue
        print(f"{name} ({r['bits_per_value']:.2f} bits):", flush=True)
        for task, sc in r["scores"].items():
            summary = ", ".join(f"{k}={v:.3f}" for k, v in sc.items())
            print(f"  {task}: {summary}", flush=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": model_name,
        "tasks": tasks,
        "seq_lens": seq_lens,
        "limit": limit,
        "results": results,
    }, indent=2))
    print(f"\nSaved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--tasks", default="ruler_qa_squad,ruler_qa_hotpot,ruler_vt",
                   help="Comma-separated RULER base task names (without _<seqlen>).")
    p.add_argument("--seq-lens", default="4096,8192",
                   help="Comma-separated context lengths to evaluate.")
    p.add_argument("--configs", default="",
                   help="Subset: fp16, int4, hqmq_s96_r6_Med3. Default = all.")
    p.add_argument("--limit", type=int, default=50,
                   help="Max examples per (task, seq_len). RULER is slow at long context.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    tasks_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seq_lens = [int(s.strip()) for s in args.seq_lens.split(",") if s.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    if args.out is None:
        ts = int(time.time())
        model_tag = args.model.replace("/", "_")
        args.out = f"runs/55_ruler_{model_tag}_{ts}.json"

    run(args.model, tasks_list, seq_lens, configs, args.limit, args.out)


if __name__ == "__main__":
    main()
