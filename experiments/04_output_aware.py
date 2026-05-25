"""Fourth experiment: output-aware (calibrated) learned rotation
on top of the spherical+JL quantizer.

Hypothesis: per-(layer, head) rotations trained to minimize attention-output
MSE on a calibration set will close the gap to fp16 at sub-3-bit budgets.
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
from src.quantizers.output_aware import LearnedRotationKVQuantizer
from src.calibration import calibrate_output_aware, sample_calibration_data
from src.eval.perplexity import wikitext_perplexity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-1b-deduped")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-windows", type=int, default=10)
    p.add_argument("--calib-seqs", type=int, default=16)
    p.add_argument("--calib-seq-len", type=int, default=512)
    p.add_argument("--calib-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--radius-bits", type=int, default=4)
    p.add_argument("--jl-dim", type=int, default=4)
    p.add_argument("--chunk-dim", type=int, default=4)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = "cuda"

    print(f"Loading {args.model} ({args.dtype}) ...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=dtype)
    print(f"  loaded in {time.time() - t0:.1f}s")

    # Discover n_layers, n_heads, d_head from config
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    d_head = cfg.hidden_size // cfg.num_attention_heads
    print(f"  n_layers={n_layers}, n_kv_heads={n_heads}, d_head={d_head}")

    # 1) Baseline: spherical + JL, no rotation, no calibration
    baseline_q = SphericalProductQuantizerJL(
        chunk_dim=args.chunk_dim, radius_bits=args.radius_bits, jl_dim=args.jl_dim
    )
    print(f"\n=== baseline (sph_r{args.radius_bits}_jl{args.jl_dim}, no rotation) ===")
    res_baseline = wikitext_perplexity(
        model, tokenizer, quantizer=baseline_q,
        seq_len=args.seq_len, max_windows=args.max_windows,
    )
    print(f"  ppl = {res_baseline['perplexity']:.3f}  ({res_baseline['bits_per_value']:.2f} bits/val)")

    # 2) Identity-init rotation (sanity check — should match baseline)
    inner_q = SphericalProductQuantizerJL(
        chunk_dim=args.chunk_dim, radius_bits=args.radius_bits, jl_dim=args.jl_dim
    )
    rotated_q = LearnedRotationKVQuantizer(
        inner_q, n_layers=n_layers, n_heads=n_heads, chunk_dim=args.chunk_dim, device=device
    )
    print(f"\n=== identity-init rotation (sanity) ===")
    res_id_rot = wikitext_perplexity(
        model, tokenizer, quantizer=rotated_q,
        seq_len=args.seq_len, max_windows=args.max_windows,
    )
    print(f"  ppl = {res_id_rot['perplexity']:.3f}  (should ~match baseline)")

    # 3) Calibrate the rotation on output-MSE
    print(f"\n=== calibrating rotation (output-aware) ===")
    print(f"  Sampling {args.calib_seqs} calibration sequences of length {args.calib_seq_len} ...")
    calib_ids = sample_calibration_data(tokenizer, n_seqs=args.calib_seqs, seq_len=args.calib_seq_len)
    t0 = time.time()
    losses = calibrate_output_aware(
        model, rotated_q, calib_ids,
        steps=args.calib_steps, lr=args.lr,
        batch_size=2, log_every=20, device=device,
    )
    print(f"  calibration done in {time.time() - t0:.1f}s. Loss {losses[0]:.6f} → {losses[-1]:.6f}")

    # 4) Evaluate calibrated rotation
    print(f"\n=== calibrated rotation (output-aware) ===")
    res_calibrated = wikitext_perplexity(
        model, tokenizer, quantizer=rotated_q,
        seq_len=args.seq_len, max_windows=args.max_windows,
    )
    print(f"  ppl = {res_calibrated['perplexity']:.3f}")

    # Summary
    print("\n=== Summary ===")
    print(f"{'config':<22s} {'bits':>6s} {'ppl':>9s}")
    for tag, r in [("sph_jl (uncalibrated)", res_baseline), ("identity-init rot", res_id_rot), ("calibrated rot", res_calibrated)]:
        print(f"{tag:<22s} {r['bits_per_value']:>6.2f} {r['perplexity']:>9.3f}")
    print(f"\nCalibration: {losses[0]:.6f} → {losses[-1]:.6f}  (≈{losses[-1]/losses[0]:.2%} of starting MSE)")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"04_output_aware_{int(time.time())}.json"
    out_path.write_text(json.dumps({
        "args": vars(args),
        "baseline": res_baseline,
        "identity_init": res_id_rot,
        "calibrated": res_calibrated,
        "calib_losses": losses,
    }, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
