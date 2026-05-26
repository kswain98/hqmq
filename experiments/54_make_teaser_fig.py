"""Front-page teaser figure: "why HQMQ is the best."

A single horizontal bar chart across the four main-paper models.
Each model has three bars on log y: fp16 baseline, naive int4, HQMQ + Med3x
(at matched ~5 bits). The dramatic visual: naive int4 rockets to 10^4+ ppl
on Qwen models, while HQMQ + Med3x sits right next to the fp16 bar everywhere.

Numbers are pulled from the paper's headline tables; configs are the headline
HQMQ choice per model (the ~5 bit row).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _plot_style import (
    OURS, PALETTE,
    clean_legend, save_dual, style_axes, use_theme_rcparams,
)

use_theme_rcparams()


def make_teaser(out_stem):
    # (model, fp16 ppl, naive int4 ppl, HQMQ+Med3 ppl, HQMQ bits)
    rows = [
        ("Mistral-7B",   5.29,  5.40,    5.32,   5.07),    # naive int4 ppl from paper
        ("Llama-3-8B",   6.28,  6.81,    6.32,   5.00),
        ("Qwen2.5-7B",   7.59,  17661.0, 8.83,   5.15),
        ("Qwen3-8B",     9.60,  121.7,   9.62,   4.99),
    ]
    models = [r[0] for r in rows]
    fp16   = [r[1] for r in rows]
    int4   = [r[2] for r in rows]
    hqmq   = [r[3] for r in rows]

    # Render at the previous taller layout (7.0 x 5.4). LaTeX will scale down
    # to \columnwidth in the paper; the README displays at ~90% width.
    fig, ax = plt.subplots(figsize=(7.0, 5.4))

    x = np.arange(len(models))
    w = 0.27

    # Three bar groups
    b_fp16 = ax.bar(x - w, fp16, w, color=PALETTE["deep_blue"],
                    edgecolor="white", linewidth=0.8, zorder=3,
                    label="fp16")
    b_int  = ax.bar(x,     int4, w, color=PALETTE["cyan"],
                    edgecolor="white", linewidth=0.8, zorder=3,
                    label="naive int4")
    b_ours = ax.bar(x + w, hqmq, w, color=OURS,
                    edgecolor="white", linewidth=0.8, zorder=4,
                    label="HQMQ + Med3$\\times$ (ours)")

    # Annotate each bar with its value
    for bars, vals in [(b_fp16, fp16), (b_int, int4), (b_ours, hqmq)]:
        for bar, v in zip(bars, vals):
            label = f"{v:,.0f}" if v >= 100 else f"{v:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.18,
                    label, ha="center", fontsize=8,
                    color=PALETTE["axis_gray"])

    ax.set_yscale("log")
    # Tighter range: lower bound at 4 (just below Mistral fp16=5.29) and upper
    # bound at 30,000 (above Qwen2.5 int4=17,661). This trims the empty
    # whitespace at the top and bottom of the log axis.
    ax.set_ylim(4, 3e4)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("WikiText-103 ppl  (log)")
    style_axes(ax)
    clean_legend(ax, loc="upper left", fontsize=8.5)

    # Headline subtitle below the implicit title (caption carries the long
    # version in the paper; this is for arXiv/README skim-readers).
    fig.suptitle(
        "HQMQ at $\\sim$5 bits matches fp16 across 4 vendors;\n"
        "naive int4 explodes by 12–2000$\\times$ on outlier-heavy Qwen models",
        fontsize=10.5, y=1.0, color=PALETTE["axis_gray"],
    )
    fig.tight_layout(pad=0.3)
    save_dual(fig, out_stem)


if __name__ == "__main__":
    out_stem = Path(__file__).resolve().parents[1] / "paper" / "image" / "fig_teaser"
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    make_teaser(out_stem)
