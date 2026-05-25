"""KV cache memory accounting.

Analytic computation of memory used by fp16 vs HQMQ at various bit budgets,
across a sweep of sequence lengths and models. Outputs the table that goes
in the paper's "compression" section.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fp16_kv_bytes(n_layers, n_kv_heads, d_head, seq_len, batch=1):
    """Bytes per fp16 KV cache for the standard transformer cache (K + V)."""
    return 2 * batch * n_layers * n_kv_heads * seq_len * d_head * 2  # 2 bytes per fp16


def hqmq_kv_bytes(n_layers, n_kv_heads, d_head, seq_len, secondary_size, radius_bits, batch=1):
    """Bytes per HQMQ-compressed cache (K + V) with given config.

    Per element bits = (log2(24*S) + radius_bits) / 4 + 16/d_head
    where:
      - log2(24*S) bits is the joint codeword index per chunk (4 elements)
      - radius_bits is the per-chunk radius quantized value
      - 16/d_head is the per-token-row fp16 scale, amortized across the row
    """
    bits_dir = math.log2(24 * secondary_size)
    bits_per_chunk = bits_dir + radius_bits
    bits_per_elem = bits_per_chunk / 4 + 16 / d_head
    total_bits = 2 * batch * n_layers * n_kv_heads * seq_len * d_head * bits_per_elem
    return total_bits / 8


MODELS = {
    "Pythia-1B":   {"n_layers": 16, "n_kv_heads": 8,  "d_head": 256},
    "Pythia-2.8B": {"n_layers": 32, "n_kv_heads": 16, "d_head": 160},
    "Mistral-7B":  {"n_layers": 32, "n_kv_heads": 8,  "d_head": 128},
    "Qwen2.5-7B":  {"n_layers": 28, "n_kv_heads": 4,  "d_head": 128},
    "Llama-3-8B":  {"n_layers": 32, "n_kv_heads": 8,  "d_head": 128},
    "Llama-3-70B": {"n_layers": 80, "n_kv_heads": 8,  "d_head": 128},
}

HQMQ_CONFIGS = [
    ("hqmq_s24_r3",  {"secondary_size": 24,  "radius_bits": 3}),
    ("hqmq_s24_r4",  {"secondary_size": 24,  "radius_bits": 4}),
    ("hqmq_s48_r4",  {"secondary_size": 48,  "radius_bits": 4}),
    ("hqmq_s96_r4",  {"secondary_size": 96,  "radius_bits": 4}),
    ("hqmq_s192_r4", {"secondary_size": 192, "radius_bits": 4}),
    ("hqmq_s192_r6", {"secondary_size": 192, "radius_bits": 6}),
]

SEQ_LENS = [2048, 4096, 8192, 32768, 131072]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    args = p.parse_args()

    table = []
    for model_name, arch in MODELS.items():
        for seq_len in SEQ_LENS:
            fp16_bytes = fp16_kv_bytes(seq_len=seq_len, batch=args.batch, **arch)
            row = {
                "model": model_name,
                "seq_len": seq_len,
                "fp16_MB": fp16_bytes / 1e6,
            }
            for cfg_name, cfg in HQMQ_CONFIGS:
                hqmq_bytes = hqmq_kv_bytes(seq_len=seq_len, batch=args.batch, **arch, **cfg)
                row[f"{cfg_name}_MB"] = hqmq_bytes / 1e6
                row[f"{cfg_name}_ratio"] = fp16_bytes / hqmq_bytes
            table.append(row)

    # Print a focused table for the paper
    print(f"\n{'model':<12s} {'seq_len':>8s} {'fp16 (MB)':>10s}", end="")
    for cfg_name, _ in HQMQ_CONFIGS:
        print(f" {cfg_name+' (MB)':>16s}", end="")
    print()
    for row in table:
        if row["model"] not in ("Mistral-7B", "Qwen2.5-7B", "Llama-3-8B", "Llama-3-70B"):
            continue
        print(f"{row['model']:<12s} {row['seq_len']:>8d} {row['fp16_MB']:>10.1f}", end="")
        for cfg_name, _ in HQMQ_CONFIGS:
            mb = row[f"{cfg_name}_MB"]
            print(f" {mb:>16.1f}", end="")
        print()

    # Compression ratio table
    print(f"\n--- Compression ratio (fp16 / HQMQ) ---\n")
    print(f"{'model':<12s} {'seq_len':>8s}", end="")
    for cfg_name, _ in HQMQ_CONFIGS:
        print(f" {cfg_name+' ×':>16s}", end="")
    print()
    for row in table:
        if row["model"] not in ("Mistral-7B", "Qwen2.5-7B", "Llama-3-8B", "Llama-3-70B"):
            continue
        print(f"{row['model']:<12s} {row['seq_len']:>8d}", end="")
        for cfg_name, _ in HQMQ_CONFIGS:
            r = row[f"{cfg_name}_ratio"]
            print(f" {r:>16.2f}", end="")
        print()

    # Headline savings for the paper abstract
    print(f"\n--- Headline savings ---")
    for model_name in ["Mistral-7B", "Llama-3-8B", "Llama-3-70B"]:
        for seq_len in [8192, 32768]:
            for cfg_name, cfg in [("hqmq_s24_r3", HQMQ_CONFIGS[0][1]),
                                   ("hqmq_s96_r4", HQMQ_CONFIGS[3][1])]:
                fp = fp16_kv_bytes(seq_len=seq_len, batch=args.batch, **MODELS[model_name])
                hq = hqmq_kv_bytes(seq_len=seq_len, batch=args.batch, **MODELS[model_name], **cfg)
                print(f"  {model_name} @ {seq_len}: fp16 = {fp/1e6:.1f} MB → {cfg_name} = {hq/1e6:.1f} MB ({fp/hq:.2f}× smaller)")

    out_dir = Path(__file__).resolve().parents[1] / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"18_memory_accounting.json"
    out_path.write_text(json.dumps({"args": vars(args), "table": table}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
