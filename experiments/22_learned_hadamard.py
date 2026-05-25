"""Learned per-(layer, head) rotation matrix calibrated on Qwen2.5-7B's K/V dist.

Fixed Hadamard partially helped Qwen2.5 (rotation spreads outlier mass) but
didn't fully rescue HQMQ. Learning the rotation matrix on a small calibration
set should outperform a fixed Hadamard if Qwen's outlier pattern has structure
the calibration data can capture.

Architecture:
  - Outer: LearnedRotationKVQuantizer (chunk_dim = d_head, i.e. full-vector rotation)
  - Inner: HQMQQuantizer
  - Calibration: output-MSE on small Qwen calibration set, soft codebook + STE

Memory: per (layer, head, K|V) we have d_head*(d_head-1)/2 = 8128 skew params for d_head=128.
For Qwen (28 layers × 4 KV heads × 2): 28*4*2*8128 ≈ 1.8M params. Trainable.
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
from src.quantizers.hadamard import HadamardKVQuantizer
from src.quantizers.output_aware import LearnedRotationKVQuantizer
from src.calibration import calibrate_output_aware, sample_calibration_data
from src.eval.perplexity import wikitext_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-windows", type=int, default=10)
    p.add_argument("--calib-seqs", type=int, default=16)
    p.add_argument("--calib-seq-len", type=int, default=256)
    p.add_argument("--calib-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-3)
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

    results = []
    r = wikitext_perplexity(model, tokenizer, None, seq_len=args.seq_len, max_windows=args.max_windows)
    print(f"\nfp16 ppl = {r['perplexity']:.3f}")
    fp16_ppl = r["perplexity"]
    results.append({"name": "fp16", **r})

    # Baselines: best fixed Hadamard config and best no-Hadamard config
    print(f"\n=== Reference baselines ===")
    inner_a = HQMQQuantizer(n_layers, n_heads, secondary_size=192, radius_bits=6, device=device)
    q_a = HadamardKVQuantizer(inner_a, d_head=d_head, device=device)
    r = wikitext_perplexity(model, tokenizer, q_a, seq_len=args.seq_len, max_windows=args.max_windows)
    print(f"  Had + hqmq_s192_r6 (fixed): ppl = {r['perplexity']:.3f}")
    results.append({"name": "fixed_Had_s192_r6", **r})

    # The key experiment: LEARNED rotation + HQMQ
    print(f"\n=== Learned rotation calibrated on Qwen ===")
    inner_b = HQMQQuantizer(n_layers, n_heads, secondary_size=192, radius_bits=6, device=device)
    learned_q = LearnedRotationKVQuantizer(
        inner_b, n_layers=n_layers, n_heads=n_heads, chunk_dim=d_head, device=device
    )
    # Identity-init sanity
    print(f"  Identity-init eval (sanity, should ~match no-rotation):")
    r = wikitext_perplexity(model, tokenizer, learned_q, seq_len=args.seq_len, max_windows=args.max_windows)
    print(f"    ppl = {r['perplexity']:.3f}")
    results.append({"name": "learned_rot_id_init_s192_r6", **r})

    print(f"\n  Calibrating ({args.calib_seqs} seqs × {args.calib_seq_len} tok × {args.calib_steps} steps)...")
    calib_ids = sample_calibration_data(tokenizer, n_seqs=args.calib_seqs, seq_len=args.calib_seq_len)
    t0 = time.time()
    losses = calibrate_output_aware(
        model, learned_q, calib_ids,
        steps=args.calib_steps, lr=args.lr, batch_size=2, log_every=20, device=device,
    )
    print(f"  done in {time.time()-t0:.1f}s. loss {losses[0]:.4f} → {losses[-1]:.4f}")

    print(f"\n  Calibrated eval:")
    r = wikitext_perplexity(model, tokenizer, learned_q, seq_len=args.seq_len, max_windows=args.max_windows)
    print(f"    ppl = {r['perplexity']:.3f}")
    results.append({"name": "learned_rot_calibrated_s192_r6", **r})

    # Also try with smaller HQMQ to see if learned rotation rescues low-bit
    print(f"\n=== Learned rotation + smaller HQMQ (s24_r4) ===")
    inner_c = HQMQQuantizer(n_layers, n_heads, secondary_size=24, radius_bits=4, device=device)
    learned_c = LearnedRotationKVQuantizer(inner_c, n_layers=n_layers, n_heads=n_heads, chunk_dim=d_head, device=device)
    losses_c = calibrate_output_aware(
        model, learned_c, calib_ids,
        steps=args.calib_steps, lr=args.lr, batch_size=2, log_every=20, device=device,
    )
    print(f"  done. loss {losses_c[0]:.4f} → {losses_c[-1]:.4f}")
    r = wikitext_perplexity(model, tokenizer, learned_c, seq_len=args.seq_len, max_windows=args.max_windows)
    print(f"  calibrated ppl = {r['perplexity']:.3f}")
    results.append({"name": "learned_rot_calibrated_s24_r4", **r})

    print("\n=== Summary ===")
    print(f"{'config':<36s} {'bits':>6s} {'ppl':>9s} {'Δ vs fp16':>11s}")
    for r in results:
        print(f"{r['name']:<36s} {r['bits_per_value']:>6.2f} {r['perplexity']:>9.3f} {r['perplexity']-fp16_ppl:>+11.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"22_learned_hadamard_{args.model.split('/')[-1].replace('-','_')}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
