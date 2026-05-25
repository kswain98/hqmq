"""Generate the fused-kernel latency figure (Figure 5).

Shows decode-step latency vs context length for:
  - fp16 SDPA (cuDNN FlashAttention) on dense KV
  - Decode-then-SDPA (fake-quant pipeline)
  - Fused HQMQ-attention (Triton)

Numbers come from experiments/44_fused_attn_test.py output.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from _plot_style import (
    OURS, PALETTE,
    clean_legend, draw_foreground, save_dual, style_axes,
    use_theme_rcparams,
)

use_theme_rcparams()


def make_kernel_fig(out_stem):
    # Measured data from 44_fused_attn_test.py on RTX 4090 (current build).
    Ts        = [4096, 16384, 32768]
    sdpa      = [0.013, 0.014, 0.014]   # ms
    fakequant = [0.243, 1.389, 2.737]   # ms
    fused     = [0.033, 0.033, 0.033]   # ms

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # ---- Left: latency-vs-context, log scale ----
    # 3 series → 3 cool theme colors + amber accent for OURS.
    ax = axes[0]
    draw_foreground(ax, Ts, sdpa,
                    color=PALETTE["cyan"], marker="s", markersize=6,
                    linewidth=2.0, label="fp16 SDPA (cuDNN FlashAttention, dense K/V)",
                    smooth=False)
    draw_foreground(ax, Ts, fakequant,
                    color=PALETTE["deep_blue"], marker="x", markersize=8,
                    linewidth=2.0, label="Decode-then-SDPA (fake-quant pipeline)",
                    smooth=False)
    draw_foreground(ax, Ts, fused,
                    color=OURS, marker="o", markersize=7,
                    linewidth=2.6, label="Fused HQMQ-Attention (ours)",
                    is_ours=True, smooth=False)

    ax.set_xlabel(r"KV-cache context length $T_{kv}$ (tokens)")
    ax.set_ylabel("Decode-step latency (ms/call)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ts)
    ax.set_xticklabels([f"{t//1024}k" for t in Ts])
    ax.set_title("Decode-step latency vs context length (Mistral-class GQA)")
    style_axes(ax)
    clean_legend(ax, loc="upper left")

    # ---- Right: speedup bars ----
    ax = axes[1]
    speedups = [fq / fu for fq, fu in zip(fakequant, fused)]
    labels = [f"{t//1024}k" for t in Ts]
    bars = ax.bar(labels, speedups, color=OURS, edgecolor="white", linewidth=0.8,
                  zorder=3)
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}x", ha="center", fontsize=10,
                color=PALETTE["axis_gray"], fontweight="bold")
    ax.set_ylabel(r"Overhead of not fusing decode ($\times$)")
    ax.set_xlabel(r"Context length $T_{kv}$")
    ax.set_title("Cost of not fusing decode into attention")
    ax.set_ylim(0, max(speedups) * 1.2)
    style_axes(ax)

    fig.suptitle("Fused HQMQ-Attention kernel vs fp16 SDPA upper bound (RTX 4090)",
                 fontsize=12, y=1.02, color=PALETTE["axis_gray"])
    fig.tight_layout()
    save_dual(fig, out_stem)


if __name__ == "__main__":
    out_stem = Path(__file__).resolve().parents[1] / "paper" / "image" / "fig5_kernel"
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    make_kernel_fig(out_stem)
