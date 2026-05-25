"""Outlier-rate figure (Figure 6).

For each layer of each model, plot the per-head K-chunk max/median ratio.
This visualizes why Qwen-class architectures need Med3x extraction and
Mistral-class don't.

Data source: runs/23_kv_diagnosis.json (already gathered for Mistral + Qwen2.5).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _plot_style import (
    PALETTE,
    clean_legend, draw_foreground, save_dual, style_axes,
    use_theme_rcparams,
)

use_theme_rcparams()


MODEL_COLOR = {
    "Mistral-7B":  PALETTE["deep_blue"],
    "Qwen2.5-7B":  PALETTE["amber"],     # outlier-heavy model = headline accent
}


def _short_name(model_name):
    return "Mistral-7B" if "Mistral" in model_name else "Qwen2.5-7B"


def make_outlier_fig(out_stem, runs_root):
    diag_path = runs_root / "23_kv_diagnosis.json"
    with open(diag_path) as f:
        diag = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # ---- Left: per-layer max/median ratio per model ----
    ax = axes[0]
    for model_name, layers in diag.items():
        short = _short_name(model_name)
        per_layer = []
        for layer_idx in sorted(layers.keys(), key=int):
            K = layers[layer_idx]["K"]
            per_layer.append(K["per_head_max_to_median"])
        arr = np.array(per_layer)  # (n_layers, n_kv_heads)
        n_layers = arr.shape[0]
        mn = arr.min(axis=1)
        mx = arr.max(axis=1)
        layers_idx = np.arange(n_layers)
        ax.fill_between(layers_idx, mn, mx,
                        color=MODEL_COLOR[short], alpha=0.18, zorder=2)
        ax.plot(layers_idx, mx, "-", color=MODEL_COLOR[short],
                linewidth=2.2 if short == "Qwen2.5-7B" else 1.8,
                marker="o" if short == "Qwen2.5-7B" else "s",
                markersize=4.5, markerfacecolor=MODEL_COLOR[short],
                markeredgecolor="white", markeredgewidth=0.6,
                zorder=4 if short == "Qwen2.5-7B" else 3,
                label=f"{short} max-head")

    ax.axhline(3, ls=(0, (5, 3)), color=PALETTE["axis_gray"], alpha=1.0,
               linewidth=1.4, zorder=1, label=r"$C=3$ Med3$\times$ threshold")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("K-chunk norm:  max / median")
    ax.set_yscale("log")
    ax.set_title("Per-layer K-chunk outlier ratio (max-head, range across heads)")
    style_axes(ax)
    clean_legend(ax, loc="upper left")

    # ---- Right: chunk-norm quantile profile ----
    # Plot in "complement-quantile" log space (x = 1 - q) and invert the axis.
    # The conventional way to render distribution tails: high quantiles (the
    # outliers we care about) get equal x-space instead of being crushed into
    # the right edge as they would on a plain log(q) axis.
    ax = axes[1]
    qs = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    xs_complement = [1.0 - q for q in qs]      # 0.99, 0.9, 0.5, 0.1, 0.01, 0.001
    for model_name, layers in diag.items():
        short = _short_name(model_name)
        per_layer_quants = []
        for layer_idx in sorted(layers.keys(), key=int):
            K = layers[layer_idx]["K"]
            per_layer_quants.append(K["chunk_norm_quantiles"])
        median_quant = np.median(np.array(per_layer_quants), axis=0)
        draw_foreground(
            ax, xs_complement, median_quant,
            color=MODEL_COLOR[short],
            marker="o" if short == "Qwen2.5-7B" else "s",
            markersize=6,
            linewidth=2.4 if short == "Qwen2.5-7B" else 2.0,
            label=short,
            is_ours=(short == "Qwen2.5-7B"),
            smooth=True, log_x=True,
        )

    ax.set_xlabel("Quantile of K-chunk norm distribution")
    ax.set_ylabel("Chunk norm (median across layers)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xs_complement)
    ax.set_xticklabels(["1%", "10%", "50%", "90%", "99%", "99.9%"])
    ax.invert_xaxis()    # so low quantiles (bulk) sit on the left, tail on the right
    ax.set_title(r"K-chunk norm tail (Qwen has 50$\times$ wider tail than Mistral)")
    style_axes(ax)
    clean_legend(ax, loc="upper left", fontsize=10)

    fig.suptitle("Why Qwen-class models need outlier extraction (and Mistral-class don't)",
                 fontsize=12, y=1.02, color=PALETTE["axis_gray"])
    fig.tight_layout()
    save_dual(fig, out_stem)


if __name__ == "__main__":
    runs_root = Path(__file__).resolve().parents[1] / "runs"
    out_stem = Path(__file__).resolve().parents[1] / "paper" / "image" / "fig6_outlier_diag"
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    make_outlier_fig(out_stem, runs_root)
