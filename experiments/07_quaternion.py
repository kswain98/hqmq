"""Quaternion-parameterized rotation calibration on sph+JL.

The rotation R per (layer, head) is parameterized as a unit quaternion (4 params),
giving the left-isoclinic subgroup of SO(4). Cleaner than 6-param skew matrix
+ matrix_exp; native unit-norm constraint should stabilize calibration.

Also makes the entire pipeline algebraically quaternionic — the 24-cell codebook
is itself the unit Hurwitz quaternion group, so quaternion rotation composes
naturally with quaternion codeword selection.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_utils import load_model
from src.quantizers.spherical import SphericalProductQuantizerJL
from src.quantizers.output_aware import LearnedRotationKVQuantizer, QuaternionRotationKVQuantizer
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
    print(f"  ppl = {res['perplexity']:.3f}")
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
    p.add_argument("--radius-bits", type=int, default=4)
    p.add_argument("--jl-dim", type=int, default=4)
    p.add_argument("--chunk-dim", type=int, default=4)
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

    def fresh_inner():
        return SphericalProductQuantizerJL(
            chunk_dim=args.chunk_dim, radius_bits=args.radius_bits, jl_dim=args.jl_dim
        )

    results = {}
    # 1) Baseline
    results["baseline"] = run_one("baseline (no rotation)", fresh_inner(), model, tokenizer, args)

    # 2) Skew-matrix rotation (existing, full SO(4) — 6 params per layer-head)
    skew_q = LearnedRotationKVQuantizer(
        fresh_inner(), n_layers=n_layers, n_heads=n_heads, chunk_dim=args.chunk_dim, device=device
    )
    results["skew_calibrated"] = run_one(
        "skew rotation calibrated (6 params)",
        skew_q, model, tokenizer, args, calib_ids=calib_ids, do_calibrate=True,
    )

    # 3) Quaternion rotation (4 params per layer-head)
    quat_q = QuaternionRotationKVQuantizer(
        fresh_inner(), n_layers=n_layers, n_heads=n_heads, chunk_dim=args.chunk_dim, device=device
    )
    results["quat_calibrated"] = run_one(
        "quaternion rotation calibrated (4 params)",
        quat_q, model, tokenizer, args, calib_ids=calib_ids, do_calibrate=True,
    )

    # Summary
    print("\n=== Summary ===")
    print(f"{'config':<40s} {'bits':>6s} {'ppl':>9s}")
    for tag in ["baseline", "skew_calibrated", "quat_calibrated"]:
        r = results[tag]
        print(f"{tag:<40s} {r['bits_per_value']:>6.2f} {r['perplexity']:>9.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"07_quaternion_{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "results": {k: v for k, v in results.items()}}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
