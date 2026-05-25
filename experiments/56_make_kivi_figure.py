"""Generate fig_kivi_head2head from runs/53_kivi_compare_*.json.

Three-panel grouped bar chart: CoQA EM, TruthfulQA bleu_max, GSM8K exact match.
Each panel shows 5 configs ordered by bit budget:
    KIVI-2 (~2.5 b) ─ HQMQ s24_r2 (2.79) ─ HQMQ s96_r4 (3.79)
                    ─ HQMQ s96_r4 + Med3 (4.41) ─ KIVI-4 (~4.5)
plus an fp16 dashed reference line (averaged over the two fp16 runs).

The visual story is the 3.79-bit HQMQ bar matching the 4.5-bit KIVI-4 bar at
16% fewer bits, and HQMQ + Med3 crossing KIVI-4 on CoQA.
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_style import (
    PALETTE, OURS, FP16,
    style_axes, draw_reference_hline, clean_legend,
    use_theme_rcparams, save_dual,
)

use_theme_rcparams()


# ---------------------------------------------------------------------------
# KIVI paper Table 3 (Mistral-7B, arXiv:2402.02750)
# ---------------------------------------------------------------------------
KIVI_PUBLISHED = {
    "fp16": {"bits": 16.0, "coqa": 67.40, "tqa": 30.45, "gsm8k": 38.36},
    "KIVI-4 (calib.)": {"bits": 4.5, "coqa": 66.95, "tqa": 30.49, "gsm8k": 37.30},
    "KIVI-2 (calib.)": {"bits": 2.5, "coqa": 66.35, "tqa": 32.17, "gsm8k": 36.01},
}


def load_our_runs():
    runs_root = Path(__file__).resolve().parents[1] / "runs"
    matches = sorted(glob.glob(str(runs_root / "53_kivi_compare_*.json")))
    if not matches:
        raise FileNotFoundError(
            "no KIVI compare data found in runs/53_kivi_compare_*.json"
        )
    return json.loads(Path(matches[-1]).read_text(encoding="utf-8"))


def extract_scores(run):
    """Pull CoQA EM, TruthfulQA bleu_max, GSM8K exact_match into a flat dict."""
    out = {}
    for cfg_name, cfg_data in run["results"].items():
        if "error" in cfg_data:
            continue
        sc = cfg_data["scores"]
        out[cfg_name] = {
            "bits": cfg_data["bits_per_value"],
            "coqa":  100 * sc["coqa"]["em,none"],
            "tqa":   sc["truthfulqa_gen"]["bleu_max,none"],
            "gsm8k": 100 * sc["gsm8k"]["exact_match,strict-match"],
        }
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def main():
    run = load_our_runs()
    ours = extract_scores(run)

    # Configs to show, in bit-ascending order. KIVI rows come from the paper.
    rows = [
        # (label, bits, color, scores_dict)
        ("KIVI-2 (calib.)",      KIVI_PUBLISHED["KIVI-2 (calib.)"]["bits"],
         PALETTE["med_blue"],    KIVI_PUBLISHED["KIVI-2 (calib.)"]),
        ("HQMQ s24_r2",          ours["hqmq_s24_r2"]["bits"],
         PALETTE["teal"],        ours["hqmq_s24_r2"]),
        ("HQMQ s96_r4 (ours)",   ours["hqmq_s96_r4"]["bits"],
         OURS,                   ours["hqmq_s96_r4"]),
        ("HQMQ s96_r4 + Med3x",  ours["hqmq_s96_r4_Med3"]["bits"],
         PALETTE["deep_blue"],   ours["hqmq_s96_r4_Med3"]),
        ("KIVI-4 (calib.)",      KIVI_PUBLISHED["KIVI-4 (calib.)"]["bits"],
         PALETTE["med_blue"],    KIVI_PUBLISHED["KIVI-4 (calib.)"]),
    ]

    tasks = [("coqa",  "CoQA EM"),
             ("tqa",   "TruthfulQA BLEU"),
             ("gsm8k", "GSM8K exact match")]

    # fp16 reference per panel (averaged across the two fp16 baselines we have:
    # KIVI's published 16-bit and our own n=200 run).
    fp16_ref = {
        "coqa":  (KIVI_PUBLISHED["fp16"]["coqa"]  + ours["fp16"]["coqa"])  / 2,
        "tqa":   (KIVI_PUBLISHED["fp16"]["tqa"]   + ours["fp16"]["tqa"])   / 2,
        "gsm8k": (KIVI_PUBLISHED["fp16"]["gsm8k"] + ours["fp16"]["gsm8k"]) / 2,
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    x = np.arange(len(rows))
    width = 0.66

    for ax, (key, title) in zip(axes, tasks):
        scores = [r[3][key] for r in rows]
        colors = [r[2] for r in rows]
        bars = ax.bar(x, scores, width, color=colors,
                      edgecolor="white", linewidth=0.8, zorder=3)
        # Annotate each bar with its score
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                    f"{score:.1f}", ha="center", va="bottom",
                    fontsize=8, color=PALETTE["axis_gray"], zorder=4)
        draw_reference_hline(ax, fp16_ref[key],
                             label=f"fp16 ({fp16_ref[key]:.1f})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r[1]:.1f}b" for r in rows],
                           fontsize=8.5)
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(scores + [fp16_ref[key]]) * 1.18)
        style_axes(ax)
        ax.legend(loc="lower right", fontsize=8, frameon=False)

    # Build a single legend at the top for config families.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["med_blue"],
                      label="KIVI (calibrated)"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["teal"],
                      label="HQMQ sub-3-bit (no calibration)"),
        plt.Rectangle((0, 0), 1, 1, color=OURS,
                      label="HQMQ s96_r4 @ 3.79b (ours, no calibration)"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["deep_blue"],
                      label="HQMQ s96_r4 + Med3$\\times$ @ 4.41b"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), ncol=4,
               frameon=False, fontsize=8.5)

    fig.tight_layout()
    out_dir = Path(__file__).resolve().parents[1] / "paper" / "image"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dual(fig, out_dir / "fig_kivi_head2head")


if __name__ == "__main__":
    main()
