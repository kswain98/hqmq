"""Test Hurwitz Quaternion Multiplicative Quantizer (HQMQ).

Compares against:
  - fp16 baseline
  - Naive int3, int4
  - sph_r4_jl4 (our previous best at this bit budget) — uncalibrated and calibrated

HQMQ configs span (secondary_size, radius_bits) to map the bit/quality curve.
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
from src.quantizers.spherical import SphericalProductQuantizerJL
from src.quantizers.hqmq import HQMQQuantizer
from src.calibration import calibrate_output_aware, sample_calibration_data
from src.eval.perplexity import wikitext_perplexity


def run_one(name, quantizer, model, tokenizer, args, calib_ids=None, do_calibrate=False):
    print(f"\n=== {name} ===")
    if do_calibrate and calib_ids is not None:
        t0 = time.time()
        losses = calibrate_output_aware(
            model, quantizer, calib_ids,
            steps=args.calib_steps, lr=args.lr,
            batch_size=2, log_every=50, device="cuda",
        )
        print(f"  calibrated in {time.time() - t0:.1f}s. Loss {losses[0]:.4f} → {losses[-1]:.4f}")
    res = wikitext_perplexity(
        model, tokenizer, quantizer=quantizer,
        seq_len=args.seq_len, max_windows=args.max_windows,
    )
    print(f"  ppl = {res['perplexity']:.3f}  ({res['bits_per_value']:.2f} bits/val)")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-1b-deduped")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-windows", type=int, default=10)
    p.add_argument("--calib-seqs", type=int, default=16)
    p.add_argument("--calib-seq-len", type=int, default=512)
    p.add_argument("--calib-steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = "cuda"
    print(f"Loading {args.model} ({args.dtype}) ...")
    model, tokenizer = load_model(args.model, dtype=dtype)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    d_head = cfg.hidden_size // cfg.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv_heads={n_heads}, d_head={d_head}")

    calib_ids = sample_calibration_data(tokenizer, n_seqs=args.calib_seqs, seq_len=args.calib_seq_len)
    results = []

    # 1) fp16
    results.append(("fp16", run_one("fp16", None, model, tokenizer, args)))

    # 2) Naive int baselines
    results.append(("int4", run_one("int4", NaivePerTokenIntQuantizer(bits=4), model, tokenizer, args)))
    results.append(("int3", run_one("int3", NaivePerTokenIntQuantizer(bits=3), model, tokenizer, args)))

    # 3) Existing best (sph_r4_jl4, uncalibrated)
    sph = SphericalProductQuantizerJL(chunk_dim=4, radius_bits=4, jl_dim=4)
    results.append(("sph_r4_jl4_uncal", run_one("sph_r4_jl4 (uncalibrated)", sph, model, tokenizer, args)))

    # 4) HQMQ variants — uncalibrated first, to map the codebook-size effect
    for s_size in [24, 48]:
        for r_bits in [4, 2]:
            name = f"hqmq_s{s_size}_r{r_bits}_uncal"
            hqmq = HQMQQuantizer(
                n_layers=n_layers, n_heads=n_heads,
                secondary_size=s_size, radius_bits=r_bits,
                device=device, init="random",
            )
            results.append((name, run_one(name, hqmq, model, tokenizer, args)))

    # 5) HQMQ calibrated — the headline test
    name = "hqmq_s24_r4_calibrated"
    hqmq_cal = HQMQQuantizer(
        n_layers=n_layers, n_heads=n_heads,
        secondary_size=24, radius_bits=4,
        device=device, init="random",
    )
    results.append((name, run_one(name, hqmq_cal, model, tokenizer, args,
                                   calib_ids=calib_ids, do_calibrate=True)))

    name = "hqmq_s24_r2_calibrated"
    hqmq_cal2 = HQMQQuantizer(
        n_layers=n_layers, n_heads=n_heads,
        secondary_size=24, radius_bits=2,
        device=device, init="random",
    )
    results.append((name, run_one(name, hqmq_cal2, model, tokenizer, args,
                                   calib_ids=calib_ids, do_calibrate=True)))

    print("\n=== Summary (sorted by bits) ===")
    fp16_ppl = next(r["perplexity"] for _, r in results if r["bits_per_value"] == 16.0)
    sorted_r = sorted(results, key=lambda x: x[1]["bits_per_value"])
    print(f"{'config':<28s} {'bits':>6s} {'ppl':>9s} {'Δ vs fp16':>11s}")
    for name, r in sorted_r:
        delta = r["perplexity"] - fp16_ppl
        print(f"{name:<28s} {r['bits_per_value']:>6.2f} {r['perplexity']:>9.3f} {delta:>+11.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"08_hqmq_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": [{"name": n, **r} for n, r in results]}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
