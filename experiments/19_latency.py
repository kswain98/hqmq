"""Per-token generation latency: fp16 vs QuantizedCache.

Measures the wall-clock overhead introduced by the quantize/dequantize step.
In real deployment with a custom int4 attention kernel, this overhead would
be replaced by direct int-codebook attention, which is generally faster
than fp16. Our measurement here gives a *conservative upper bound* on the
extra compute cost — production deployment would be faster than what we
measure, because we currently dequantize-to-fp16-then-do-fp16-attention.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import DynamicCache

from src.model_utils import load_model, QuantizedCache
from src.quantizers.naive_int4 import NaivePerTokenIntQuantizer
from src.quantizers.hqmq import HQMQQuantizer


@torch.no_grad()
def measure_latency(model, tokenizer, quantizer, prompt_len: int, gen_len: int,
                    *, warmup: int = 2, n_runs: int = 5, device: str = "cuda"):
    """Generate gen_len tokens after a prompt of prompt_len. Time the full pass.

    Returns mean tokens/sec and total wall time per run.
    """
    # Build a fixed prompt of approximately prompt_len tokens (use random ids in vocab range).
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size - 1, (1, prompt_len), device=device)

    times = []
    for run in range(warmup + n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache = QuantizedCache(quantizer) if quantizer is not None else DynamicCache()
        # Prefill
        out = model(input_ids, past_key_values=cache, use_cache=True)
        # Greedy decode gen_len tokens
        next_id = out.logits[:, -1:].argmax(-1)
        for _ in range(gen_len - 1):
            out = model(next_id, past_key_values=cache, use_cache=True)
            next_id = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if run >= warmup:
            times.append(t1 - t0)

    mean = sum(times) / len(times)
    return {
        "mean_wall_time_s": mean,
        "tokens_per_sec": (prompt_len + gen_len) / mean,
        "decode_tok_per_sec": gen_len / mean,
        "n_runs": n_runs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--prompt-lens", type=int, nargs="+", default=[1024, 4096])
    p.add_argument("--gen-len", type=int, default=32)
    p.add_argument("--n-runs", type=int, default=3)
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
        ("hqmq_s24_r3", HQMQQuantizer(n_layers, n_heads, secondary_size=24, radius_bits=3, device=device)),
        ("hqmq_s96_r4", HQMQQuantizer(n_layers, n_heads, secondary_size=96, radius_bits=4, device=device)),
        ("hqmq_s192_r4", HQMQQuantizer(n_layers, n_heads, secondary_size=192, radius_bits=4, device=device)),
    ]

    results = []
    for prompt_len in args.prompt_lens:
        print(f"\n========== prompt_len = {prompt_len}, gen_len = {args.gen_len} ==========")
        for name, q in configs:
            print(f"  {name} ...", end=" ", flush=True)
            r = measure_latency(model, tokenizer, q, prompt_len, args.gen_len, n_runs=args.n_runs, device=device)
            bits = q.bits_per_value() if q is not None else 16.0
            print(f"decode = {r['decode_tok_per_sec']:.2f} tok/s  total = {r['mean_wall_time_s']:.2f}s  ({bits:.2f} bits)")
            results.append({"name": name, "bits": bits, "prompt_len": prompt_len, **r})

    print("\n=== Summary ===")
    print(f"{'config':<14s} {'bits':>6s} {'prompt':>8s} {'decode tok/s':>13s} {'rel speed':>10s}")
    for prompt_len in args.prompt_lens:
        fp16_speed = next(r["decode_tok_per_sec"] for r in results if r["name"] == "fp16" and r["prompt_len"] == prompt_len)
        for r in results:
            if r["prompt_len"] != prompt_len:
                continue
            rel = r["decode_tok_per_sec"] / fp16_speed
            print(f"{r['name']:<14s} {r['bits']:>6.2f} {r['prompt_len']:>8d} {r['decode_tok_per_sec']:>13.2f} {rel:>10.2%}")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1].replace("-", "_")
    out_path = out_dir / f"19_latency_{model_tag}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
