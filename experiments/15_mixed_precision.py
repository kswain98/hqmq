"""Mixed-precision HQMQ via per-layer sensitivity probing.

For each layer (and optionally each head), measure how much ppl degrades when
that single layer's KV is quantized aggressively while everything else is high
precision. Layers with high sensitivity get more bits; layers with low
sensitivity get fewer. Target: 3-bit *average* with lossless quality on
Mistral-7B.

This is the KVTuner-style move applied with HQMQ as the underlying quantizer.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from src.model_utils import load_model, QuantizedCache
from src.quantizers.hqmq import HQMQQuantizer
from src.quantizers.base import KVQuantizer
from src.eval.perplexity import wikitext_perplexity


class PerLayerHQMQQuantizer(KVQuantizer):
    """HQMQ with per-layer configurable (secondary_size, radius_bits).

    Stores one HQMQQuantizer per layer; routes by layer_idx.
    """

    def __init__(self, n_layers: int, n_heads: int, configs: List[Dict], device: str = "cuda"):
        """configs: list of dicts with keys {secondary_size, radius_bits}, one per layer."""
        assert len(configs) == n_layers
        self.n_layers = n_layers
        self.n_heads = n_heads
        self._quants = []
        for cfg in configs:
            self._quants.append(HQMQQuantizer(
                n_layers=1, n_heads=n_heads,
                secondary_size=cfg["secondary_size"],
                radius_bits=cfg["radius_bits"],
                device=device,
                seed=cfg.get("seed", 42),
            ))
        self._configs = configs

    def quantize_K(self, k, layer_idx):
        return self._quants[layer_idx].quantize_K(k, 0)

    def quantize_V(self, v, layer_idx):
        return self._quants[layer_idx].quantize_V(v, 0)

    def bits_per_value(self):
        # Average across layers.
        import math
        bits = [(math.log2(24 * c["secondary_size"]) + c["radius_bits"]) / 4.0 for c in self._configs]
        return sum(bits) / len(bits)

    @property
    def name(self):
        bit_summary = f"{self.bits_per_value():.2f}avg"
        return f"PerLayerHQMQ_{bit_summary}"


@torch.no_grad()
def sensitivity_probe(model, tokenizer, n_layers, n_heads, *,
                      seq_len: int = 1024, max_windows: int = 5, device: str = "cuda",
                      probe_secondary: int = 24, probe_radius: int = 3,
                      ref_secondary: int = 192, ref_radius: int = 6) -> List[float]:
    """For each layer, compute Δppl when that layer is at low-bit HQMQ and all
    others at high-bit HQMQ. Higher Δ = more sensitive layer.
    """
    sensitivities = [0.0] * n_layers
    # Baseline: all-high HQMQ
    high_configs = [{"secondary_size": ref_secondary, "radius_bits": ref_radius}] * n_layers
    base_q = PerLayerHQMQQuantizer(n_layers, n_heads, high_configs, device=device)
    base_r = wikitext_perplexity(model, tokenizer, base_q, seq_len=seq_len, max_windows=max_windows)
    base_ppl = base_r["perplexity"]
    print(f"  All-high baseline (s{ref_secondary}_r{ref_radius}): ppl = {base_ppl:.4f}")

    for layer_idx in range(n_layers):
        cfgs = [{"secondary_size": ref_secondary, "radius_bits": ref_radius} for _ in range(n_layers)]
        cfgs[layer_idx] = {"secondary_size": probe_secondary, "radius_bits": probe_radius}
        q = PerLayerHQMQQuantizer(n_layers, n_heads, cfgs, device=device)
        r = wikitext_perplexity(model, tokenizer, q, seq_len=seq_len, max_windows=max_windows)
        delta = r["perplexity"] - base_ppl
        sensitivities[layer_idx] = delta
        print(f"  layer {layer_idx:2d}: ppl={r['perplexity']:.4f}  Δ={delta:+.4f}")
    return sensitivities, base_ppl


def _cfg_bits(cfg):
    return (math.log2(24 * cfg["secondary_size"]) + cfg["radius_bits"]) / 4.0


def allocate_bits_by_sensitivity(sensitivities: List[float], target_avg_bits: float,
                                  available_configs: List[Dict]) -> List[Dict]:
    """Greedy: walk layers in sensitivity order (most sensitive first). Each layer
    gets the LARGEST upgrade that fits in remaining bit budget. No downgrades."""
    sorted_by_bits = sorted(available_configs, key=lambda c: -_cfg_bits(c))  # largest first
    smallest = sorted_by_bits[-1]
    n_layers = len(sensitivities)
    result = [smallest] * n_layers
    cur_total = n_layers * _cfg_bits(smallest)
    target_total = target_avg_bits * n_layers

    sort_idx = sorted(range(n_layers), key=lambda i: -sensitivities[i])
    for idx in sort_idx:
        cur_bits = _cfg_bits(result[idx])
        for cfg in sorted_by_bits:  # try largest first
            new_bits = _cfg_bits(cfg)
            if new_bits <= cur_bits:
                break  # only upgrades; configs sorted by bits desc, so we're done
            delta = new_bits - cur_bits
            if cur_total + delta <= target_total + 0.01:
                result[idx] = cfg
                cur_total += delta
                break  # take the largest fitting upgrade for this layer
    return result


def main():
    import math as _m
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-windows", type=int, default=10)
    p.add_argument("--probe-windows", type=int, default=5)
    p.add_argument("--target-avg-bits", type=float, default=3.0)
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

    print(f"\n=== fp16 baseline ===")
    r_fp16 = wikitext_perplexity(model, tokenizer, None, seq_len=args.seq_len, max_windows=args.max_windows)
    print(f"  ppl = {r_fp16['perplexity']:.3f}")
    fp16_ppl = r_fp16["perplexity"]

    print(f"\n=== Uniform HQMQ at target avg bits = {args.target_avg_bits} ===")
    # Find uniform config closest to target bits
    candidates = [
        {"secondary_size": 24, "radius_bits": 3},   # 3.04 bits
        {"secondary_size": 48, "radius_bits": 3},   # 3.29 bits
        {"secondary_size": 24, "radius_bits": 4},   # 3.29 bits
        {"secondary_size": 96, "radius_bits": 3},   # 3.54 bits
    ]
    closest = min(candidates,
                  key=lambda c: abs((_m.log2(24 * c["secondary_size"]) + c["radius_bits"]) / 4.0 - args.target_avg_bits))
    cb = (_m.log2(24 * closest["secondary_size"]) + closest["radius_bits"]) / 4.0
    print(f"  picked {closest} → {cb:.2f} bits")
    q_uniform = HQMQQuantizer(n_layers, n_heads, secondary_size=closest["secondary_size"],
                              radius_bits=closest["radius_bits"], device=device)
    r_uniform = wikitext_perplexity(model, tokenizer, q_uniform, seq_len=args.seq_len, max_windows=args.max_windows)
    print(f"  uniform ppl = {r_uniform['perplexity']:.3f}")

    print(f"\n=== Sensitivity probe (one layer downquantized at a time) ===")
    sensitivities, ref_ppl = sensitivity_probe(
        model, tokenizer, n_layers, n_heads,
        seq_len=args.seq_len, max_windows=args.probe_windows, device=device,
    )
    print(f"\n  Sensitivities (Δppl): {[round(s, 4) for s in sensitivities]}")
    print(f"  Sorted layers (most sensitive first): {sorted(range(n_layers), key=lambda i: -sensitivities[i])}")

    available_configs = [
        {"secondary_size": 24, "radius_bits": 2},   # 2.79 bits ← very cheap
        {"secondary_size": 24, "radius_bits": 3},   # 3.04
        {"secondary_size": 48, "radius_bits": 3},   # 3.29
        {"secondary_size": 24, "radius_bits": 4},   # 3.29
        {"secondary_size": 48, "radius_bits": 4},   # 3.54
        {"secondary_size": 96, "radius_bits": 4},   # 3.79
        {"secondary_size": 192, "radius_bits": 4},  # 4.04
        {"secondary_size": 192, "radius_bits": 6},  # 4.54 ← expensive
    ]
    layer_configs = allocate_bits_by_sensitivity(sensitivities, args.target_avg_bits, available_configs)
    print(f"\n=== Allocated configs (avg bits target = {args.target_avg_bits}) ===")
    for li, cfg_li in enumerate(layer_configs):
        b = (_m.log2(24 * cfg_li["secondary_size"]) + cfg_li["radius_bits"]) / 4.0
        print(f"  layer {li:2d}: s={cfg_li['secondary_size']:3d}, r={cfg_li['radius_bits']} → {b:.2f} bits  (sens={sensitivities[li]:+.4f})")

    avg_bits = sum((_m.log2(24 * c["secondary_size"]) + c["radius_bits"]) / 4.0 for c in layer_configs) / n_layers
    print(f"  avg bits = {avg_bits:.3f}")

    print(f"\n=== Mixed-precision HQMQ eval ===")
    q_mixed = PerLayerHQMQQuantizer(n_layers, n_heads, layer_configs, device=device)
    r_mixed = wikitext_perplexity(model, tokenizer, q_mixed, seq_len=args.seq_len, max_windows=args.max_windows)
    print(f"  mixed ppl = {r_mixed['perplexity']:.3f}  ({avg_bits:.2f} bits avg)")

    print("\n=== Summary ===")
    print(f"{'config':<30s} {'bits':>6s} {'ppl':>9s} {'Δ vs fp16':>11s}")
    print(f"{'fp16':<30s} {16.00:>6.2f} {fp16_ppl:>9.3f} {0.0:>+11.3f}")
    print(f"{'HQMQ uniform':<30s} {cb:>6.2f} {r_uniform['perplexity']:>9.3f} {r_uniform['perplexity']-fp16_ppl:>+11.3f}")
    print(f"{'HQMQ mixed (sens)':<30s} {avg_bits:>6.2f} {r_mixed['perplexity']:>9.3f} {r_mixed['perplexity']-fp16_ppl:>+11.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1].replace("-", "_")
    out_path = out_dir / f"15_mixed_precision_{model_tag}_{int(time.time())}.json"
    out_path.write_text(json.dumps({
        "args": vars(args),
        "fp16": r_fp16, "uniform": r_uniform, "mixed": r_mixed,
        "sensitivities": sensitivities, "layer_configs": layer_configs,
    }, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
