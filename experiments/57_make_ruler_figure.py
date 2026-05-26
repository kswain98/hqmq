"""Generate fig_ruler.png from runs/55_ruler_*.json (both 4k and 8k runs).

Three-panel grouped bar chart: SQuAD / Hotpot / VT.
Each panel shows two-by-three bars (4k and 8k for fp16 / int4 / HQMQ).

The visual story is:
  - HQMQ preserves fp16's perfect VT score at both lengths
  - HQMQ matches fp16 on SQuAD at 8k exactly (0.602 vs 0.602)
  - naive int4 collapses on every task and the gap WIDENS with context length
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
    style_axes, draw_reference_hline, use_theme_rcparams, save_dual,
)

use_theme_rcparams()


def load_ruler_run(seqlen_filter):
    """Pick the most recent RULER run whose seq_lens contains `seqlen_filter`."""
    runs_root = Path(__file__).resolve().parents[1] / "runs"
    candidates = sorted(glob.glob(str(runs_root / "55_ruler_Qwen_Qwen3-8B_*.json")))
    if not candidates:
        raise FileNotFoundError("no RULER JSON found")
    for path in reversed(candidates):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if seqlen_filter in data.get("seq_lens", []):
            return data, path
    raise FileNotFoundError(f"no RULER run with seq_len {seqlen_filter}")


def extract_scores(run, seqlen):
    """{config -> {task -> score}} at the given seq_len."""
    out = {}
    key = f"{seqlen},none"
    for cfg, data in run["results"].items():
        if "error" in data:
            continue
        sc = data["scores"]
        out[cfg] = {
            "bits":   data["bits_per_value"],
            "squad":  sc["ruler_qa_squad"].get(key, np.nan),
            "hotpot": sc["ruler_qa_hotpot"].get(key, np.nan),
            "vt":     sc["ruler_vt"].get(key, np.nan),
        }
    return out


def main():
    run_4k, p4 = load_ruler_run(4096)
    run_8k, p8 = load_ruler_run(8192)
    print(f"4k: {p4}\n8k: {p8}")
    scores_4k = extract_scores(run_4k, 4096)
    scores_8k = extract_scores(run_8k, 8192)

    configs = [
        ("fp16",                     "fp16",         PALETTE["med_blue"]),
        ("int4",                     "naive int4",   PALETTE["cyan"]),
        ("hqmq_s96_r6_Med3",         "HQMQ+Med3$\\times$ (ours)", OURS),
    ]
    tasks = [("squad",  "SQuAD"),
             ("hotpot", "HotpotQA"),
             ("vt",     "Variable Tracking")]
    seqlens = [(4096, "4k", scores_4k), (8192, "8k", scores_8k)]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)

    n_cfgs = len(configs)
    bar_w = 0.26   # narrower bars so three fit clearly per group with breathing room

    for ax_idx, (ax, (task_key, task_title)) in enumerate(zip(axes, tasks)):
        # x positions: groups of 2 (4k / 8k), 3 bars per group, centered on 0 and 1
        for i_cfg, (cfg_key, cfg_label, cfg_color) in enumerate(configs):
            heights = []
            xs = []
            for j_seq, (sl, sl_label, sl_scores) in enumerate(seqlens):
                group_center = j_seq
                offset = (i_cfg - (n_cfgs - 1) / 2) * bar_w
                xs.append(group_center + offset)
                heights.append(sl_scores.get(cfg_key, {}).get(task_key, 0.0))
            label = cfg_label if ax_idx == 0 else None
            bars = ax.bar(xs, heights, bar_w, color=cfg_color,
                          edgecolor="white", linewidth=0.8, zorder=3,
                          label=label)
            for bar, val in zip(bars, heights):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{val:.2f}",
                        ha="center", va="bottom",
                        fontsize=7.5, color=PALETTE["axis_gray"], zorder=4)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([r"$T_{kv}{=}4$K", r"$T_{kv}{=}8$K"], fontsize=9)
        ax.set_xlim(-0.55, 1.55)   # identical x-extent across panels
        ax.set_title(task_title, fontsize=11)
        ax.set_ylim(0, 1.18)
        style_axes(ax)

    # One shared y-axis label on the leftmost panel only.
    axes[0].set_ylabel("RULER score (higher is better)")

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3,
               frameon=False, fontsize=9)

    fig.tight_layout()
    out_dir = Path(__file__).resolve().parents[1] / "paper" / "image"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dual(fig, out_dir / "fig_ruler")


if __name__ == "__main__":
    main()
