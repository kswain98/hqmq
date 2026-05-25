"""HQMQ + padding sweep on gpt-oss-20b (d_head=45 padded to 48).

gpt-oss-20b: 24 layers, 8 KV heads, d_head=45 (sparse MoE, 20B total / ~3.6B active).
Standard HQMQ doesn't apply because 45 is not divisible by 4. The PaddedHQMQ
wrapper pads to d_head=48, applies HQMQ, then truncates back. Bit overhead:
48/45 - 1 ≈ 6.7% wasted on padding.

Requires kernels package + triton >= 3.4 for MXFP4 loading on 24GB GPU.
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
from src.quantizers.hqmq_padded import PaddedHQMQQuantizer
from src.quantizers.outlier_aware import OutlierAwareHQMQQuantizer
from src.eval.perplexity import wikitext_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-windows", type=int, default=10)
    args = p.parse_args()

    print(f"Loading {args.model} ...", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=torch.bfloat16)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    d_head = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv_heads={n_heads}, d_head={d_head} (padded to {((d_head+3)//4)*4})",
          flush=True)
    print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    device = "cuda"

    results = []

    def run(name, q):
        t0 = time.time()
        try:
            r = wikitext_perplexity(model, tokenizer, q, seq_len=args.seq_len,
                                    max_windows=args.max_windows)
            elapsed = time.time() - t0
            print(f"  ppl = {r['perplexity']:.3f}  ({r['bits_per_value']:.2f} bits, {elapsed:.1f}s)",
                  flush=True)
            results.append({"name": name, **r})
            return r["perplexity"]
        except Exception as e:
            print(f"  FAILED: {str(e)[:300]}", flush=True)
            results.append({"name": name, "error": str(e)[:500]})
            return None

    print("\n=== fp16 ===", flush=True)
    fp16_ppl = run("fp16", None)

    for bits in [4, 3]:
        print(f"\n=== int{bits} ===", flush=True)
        run(f"int{bits}", NaivePerTokenIntQuantizer(bits=bits))

    # Padded HQMQ
    for s, rb in [(24, 3), (48, 4), (96, 4), (192, 4)]:
        name = f"hqmq_pad_s{s}_r{rb}"
        print(f"\n=== {name} ===", flush=True)
        q = PaddedHQMQQuantizer(n_layers=n_layers, n_heads=n_heads, d_head=d_head,
                                secondary_size=s, radius_bits=rb, device=device, init="random")
        run(name, q)
        del q
        torch.cuda.empty_cache()

    # Med3x variants
    for s in [24, 96]:
        name = f"hqmq_pad_s{s}_r6_Med3"
        print(f"\n=== {name} ===", flush=True)
        inner = PaddedHQMQQuantizer(n_layers=n_layers, n_heads=n_heads, d_head=d_head,
                                    secondary_size=s, radius_bits=6, device=device, init="random")
        # Use the inner's HQMQ for the outlier wrapper (which expects an HQMQQuantizer).
        # Hack: wrap PaddedHQMQ as if it's an HQMQ; the outlier wrapper looks at
        # chunk_dim attribute and calls .quantize_K/.quantize_V. The padded wrapper
        # has both. But it asserts isinstance(inner, HQMQQuantizer) which fails.
        # Use OutlierAwareGenericQuantizer instead.
        from src.quantizers.outlier_generic import OutlierAwareGenericQuantizer
        q = OutlierAwareGenericQuantizer(inner, chunk_dim=4,
                                         threshold_mode="median_mult", median_mult=3.0)
        run(name, q)
        torch.cuda.empty_cache()

    print("\n=== Summary (sorted by bits) ===", flush=True)
    for r in sorted([r for r in results if "perplexity" in r], key=lambda x: x["bits_per_value"]):
        delta = r["perplexity"] - (fp16_ppl or 0)
        print(f"{r['name']:<28s} {r['bits_per_value']:>6.2f} {r['perplexity']:>10.3f} {delta:>+10.3f}",
              flush=True)

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"40_gptoss_padded_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
