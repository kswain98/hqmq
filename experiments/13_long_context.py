"""Long-context perplexity at 4k–16k tokens.

KV cache quantization's actual deployment regime is long context. Tests whether
HQMQ holds at much longer sequences than the 2048-token paper-grade ppl runs.
"""

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
from src.eval.long_context import long_context_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 4096, 8192])
    p.add_argument("--max-windows", type=int, default=4)
    p.add_argument("--dataset", default="wikitext-103")
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
        ("hqmq_s24_r4", HQMQQuantizer(n_layers, n_heads, secondary_size=24, radius_bits=4, device=device)),
        ("hqmq_s96_r4", HQMQQuantizer(n_layers, n_heads, secondary_size=96, radius_bits=4, device=device)),
        ("hqmq_s192_r4", HQMQQuantizer(n_layers, n_heads, secondary_size=192, radius_bits=4, device=device)),
        ("hqmq_s192_r6", HQMQQuantizer(n_layers, n_heads, secondary_size=192, radius_bits=6, device=device)),
    ]

    results = []
    for seq_len in args.seq_lens:
        print(f"\n========== seq_len = {seq_len} ==========")
        for name, q in configs:
            print(f"\n--- {name} ---")
            t0 = time.time()
            try:
                r = long_context_perplexity(
                    model, tokenizer, q,
                    seq_len=seq_len, max_windows=args.max_windows,
                    dataset=args.dataset, device=device,
                )
                print(f"  ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits, {time.time()-t0:.1f}s)")
                results.append({"name": name, **r})
            except Exception as e:
                print(f"  FAILED: {e}")

    print("\n=== Summary ===")
    print(f"{'config':<18s} {'seq_len':>8s} {'bits':>6s} {'ppl':>9s}")
    for r in results:
        print(f"{r['name']:<18s} {r['seq_len']:>8d} {r['bits_per_value']:>6.2f} {r['perplexity']:>9.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1].replace("-", "_")
    out_path = out_dir / f"13_long_context_{model_tag}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
