"""Generate paper figures from runs/*.json data.

Figures:
  1. Pareto frontier per model (bits vs ppl) — 2x2 grid
  2. Qwen2.5 downstream accuracy under KV quantization
  3. Outlier-multiplier sweep on Qwen2.5
  4. Memory savings for Llama-3-8B / 70B

All figures use the shared theme from `_plot_style.py` (amber accent for the
proposed method, frameless legends, hidden top/right spines, no grid).
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _plot_style import (
    METHOD_STYLE, OURS, PALETTE,
    apply_k_formatter, clean_legend, draw_foreground,
    draw_reference_hline, k_formatter,
    save_dual, style_axes, use_theme_rcparams,
)

use_theme_rcparams()


# ---------------------------------------------------------------------------
# Hand-curated / combined data tables (some runs predate JSON saving)
# ---------------------------------------------------------------------------

def _qwen3_data():
    """Hand-curated Qwen3-8B sweep data (run was killed before JSON save)."""
    return [
        ("fp16",                            16.00, 9.603),
        ("int4",                             4.00, 121.715),
        ("int3",                             3.00, 588.222),
        ("int2",                             2.00, 1529.4),
        ("hqmq_s24_r3",                      3.04, 44.564),
        ("hqmq_s48_r4",                      3.54, 11.427),
        ("hqmq_s96_r4",                      3.79, 10.698),
        ("hqmq_s24_r6+Med3",                 4.51, 9.701),
        ("hqmq_s96_r6+Med3",                 4.99, 9.621),
        ("int4+Med3",                        4.62, 118.977),
    ]


def _llama_combined():
    """Llama-3-8B: prefer fresh 09 sweep JSON; merge in hardcoded Med3 entries
    from the older 31_llama_s192_only run (which isn't re-run this session)."""
    runs_root = Path(__file__).resolve().parents[1] / "runs"
    out = []
    sweep_paths = sorted(glob.glob(str(runs_root / "09_hqmq_sweep_*Llama-3-8B*.json"))
                         + glob.glob(str(runs_root / "09_hqmq_sweep_*Meta-Llama-3-8B*.json"))
                         + glob.glob(str(runs_root / "09_hqmq_sweep_*Llama_3_8B*.json"))
                         + glob.glob(str(runs_root / "09_hqmq_sweep_*Meta_Llama_3_8B*.json")))
    if sweep_paths:
        d = json.loads(Path(sweep_paths[-1]).read_text(encoding="utf-8"))
        for r in d["results"]:
            out.append((r["name"], r["bits_per_value"], r["perplexity"]))
    else:
        # Fallback: paper-grade hardcoded numbers (29_llama_sweep, 50w x 2048)
        out = [
            ("fp16",         16.00, 6.278),
            ("int4",          4.00, 6.811),
            ("int3",          3.00, 16.648),
            ("int2",          2.00, 1010.6),
            ("sph_r4_jl4",    3.15, 7.396),
            ("hqmq_s24_r3",   3.04, 7.023),
            ("hqmq_s48_r4",   3.54, 6.572),
            ("hqmq_s96_r4",   3.79, 6.479),
            ("hqmq_s192_r4", 4.04, 6.387),
            ("hqmq_s192_r6", 4.54, 6.333),
        ]
    # Med3 entries come from the older 31_llama_s192_only run (kept as hardcoded
    # because Llama-3-8B isn't outlier-heavy and doesn't need Med3 re-runs).
    out.extend([
        ("hqmq_s24_r6+Med3",   4.27, 6.586),
        ("hqmq_s96_r6+Med3",   4.76, 6.367),
        ("hqmq_s192_r6+Med3",  5.00, 6.317),
    ])
    return out


def _qwen25_combined():
    """Qwen2.5-7B: combine 09 sweep (no outlier; broken) and 30 disentanglement (with Med3)."""
    runs_root = Path(__file__).resolve().parents[1] / "runs"
    out = []
    sweep_paths = sorted(glob.glob(str(runs_root / "09_hqmq_sweep_Qwen2.5_7B*.json")))
    if sweep_paths:
        d = json.loads(Path(sweep_paths[-1]).read_text(encoding="utf-8"))
        for r in d["results"]:
            out.append((r["name"], r["bits_per_value"], r["perplexity"]))
    dis_paths = sorted(glob.glob(str(runs_root / "30_disentangle_outlier_Qwen25_7B*.json")))
    if dis_paths:
        d2 = json.loads(Path(dis_paths[-1]).read_text(encoding="utf-8"))
        for r in d2["results"]:
            if "perplexity" not in r:
                continue
            nm = r["name"]
            if "Med3" in nm and "hqmq" in nm:
                out.append((nm.replace("_Med3", "+Med3"), r["bits_per_value"], r["perplexity"]))
    return out


def _classify(nm):
    """Map a config name to one of the four plotted method-groups.

    Returns None for configs we exclude from the Pareto plot — fp16 (drawn as
    horizontal reference), and int2 (which is so far off the chart it crushes
    the y-axis range; it's reported in the table but excluded from the figure).
    """
    if nm == "fp16":
        return None
    if nm in ("int2",):
        return None      # off-chart; reported in supplementary table
    if nm.startswith("int") and "Med" not in nm:
        return "naive int"
    if nm.startswith("sph_"):
        return "sph_jl (TurboQuant)"
    if nm.startswith("hqmq") and "Med" in nm:
        return "HQMQ + Med3x"
    if nm.startswith("hqmq"):
        return "HQMQ (no extract)"
    return None


# ---------------------------------------------------------------------------
# Figure 1 — Pareto frontier, 2x2 model grid
# ---------------------------------------------------------------------------

def fig1_pareto(out_dir):
    runs_root = Path(__file__).resolve().parents[1] / "runs"

    mistral_data = []
    # Pick the latest Mistral sweep JSON by mtime
    mistral_paths = sorted(glob.glob(str(runs_root / "09_hqmq_sweep_Mistral_7B*.json")))
    if mistral_paths:
        d = json.loads(Path(mistral_paths[-1]).read_text(encoding="utf-8"))
        for r in d["results"]:
            mistral_data.append((r["name"], r["bits_per_value"], r["perplexity"]))

    panels = [
        ("Mistral-7B (dense MHA)",                  mistral_data),
        ("Llama-3-8B (GQA)",                        _llama_combined()),
        ("Qwen2.5-7B (outlier-heavy GQA)",          _qwen25_combined()),
        ("Qwen3-8B (outlier-heavy GQA, latest)",    _qwen3_data()),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=False)
    axes = axes.flatten()

    for ax, (model_name, data) in zip(axes, panels):
        if not data:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, color=PALETTE["axis_gray"])
            ax.set_title(model_name)
            style_axes(ax)
            continue

        fp16_ppl = next((p for (n, _, p) in data if n == "fp16"), None)

        groups = {k: [] for k in METHOD_STYLE}
        for (nm, b, p) in data:
            grp = _classify(nm)
            if grp is not None and grp in groups:
                groups[grp].append((b, p))

        for grp, pts in groups.items():
            if not pts:
                continue
            pts.sort()
            bits, ppls = zip(*pts)
            st = METHOD_STYLE[grp]
            draw_foreground(
                ax, bits, ppls,
                color=st["color"], marker=st["marker"], markersize=st["ms"],
                linestyle=st["ls"], linewidth=st["lw"], label=grp,
                is_ours=(grp == "HQMQ + Med3x"),
                smooth=True, log_x=False,
            )

        if fp16_ppl is not None:
            draw_reference_hline(ax, fp16_ppl, label=f"fp16 ({fp16_ppl:.2f})")

        ax.set_xlabel("bits / element")
        ax.set_ylabel("WikiText-103 perplexity")
        ax.set_yscale("log")
        ax.set_title(model_name)
        ax.set_xlim(2.5, 5.5)
        style_axes(ax)
        clean_legend(ax, loc="upper right", fontsize=8)

    fig.suptitle("HQMQ vs baselines — Pareto frontier across modern open models",
                 fontsize=12, y=1.00, color=PALETTE["axis_gray"])
    fig.tight_layout()
    save_dual(fig, out_dir / "fig1_pareto")


# ---------------------------------------------------------------------------
# Figure 2 — Qwen2.5 downstream
# ---------------------------------------------------------------------------

def fig2_qwen_downstream(out_dir):
    runs_root = Path(__file__).resolve().parents[1] / "runs"
    matches = sorted(glob.glob(str(runs_root / "28_qwen_full_*.json")))
    if not matches:
        print("no Qwen full eval data; skipping fig2")
        return
    data = json.loads(Path(matches[-1]).read_text(encoding="utf-8"))
    results = data["results"]

    tasks = ["piqa", "hellaswag", "arc_easy"]
    task_labels = ["PIQA", "HellaSwag", "ARC-Easy"]

    # Each config gets one of the 5 theme colors; the headline HQMQ+Med3
    # config gets the amber OURS accent.
    bar_palette = [
        PALETTE["deep_blue"],   # fp16            (strong cool, the upper bound)
        PALETTE["cyan"],        # int4            (lightest cool, the weak baseline)
        PALETTE["teal"],        # hqmq_s24_r6_Med3
        PALETTE["med_blue"],    # hqmq_s96_r6_Med3
        OURS,                   # hqmq_s192_r6_Med3 (headline config)
    ]

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(tasks))
    width = 0.16

    for i, r in enumerate(results):
        bits = r.get("bits_observed", r.get("bits_initial", 16.0))
        scores = [r["tasks"].get(t, {}).get("acc", 0.0) for t in tasks]
        color = bar_palette[i % len(bar_palette)]
        offset = (i - len(results) / 2) * width + width / 2
        ax.bar(x + offset, scores, width, label=f"{r['name']} ({bits:.2f}b)",
               color=color, edgecolor="white", linewidth=0.8, zorder=3)

    draw_reference_hline(ax, 0.25, label="25% random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("Zero-shot accuracy")
    ax.set_title("Qwen2.5-7B: downstream accuracy under KV quantization (n=200/task)")
    style_axes(ax)
    clean_legend(ax, loc="upper right", fontsize=8)
    fig.tight_layout()
    save_dual(fig, out_dir / "fig2_qwen_downstream")


# ---------------------------------------------------------------------------
# Figure 3 — Outlier-multiplier sweep
# ---------------------------------------------------------------------------

def fig3_outlier_sweep(out_dir):
    runs_root = Path(__file__).resolve().parents[1] / "runs"
    matches = sorted(glob.glob(str(runs_root / "26_outlier_threshold_Qwen2.5_7B_*.json")))
    if not matches:
        print("no outlier threshold sweep; skipping fig3")
        return
    data = json.loads(Path(matches[-1]).read_text(encoding="utf-8"))
    results = data["results"]

    fp16_ppl = next((r["perplexity"] for r in results if r["name"] == "fp16"), None)
    if fp16_ppl is None:
        print("no fp16 baseline; skipping fig3")
        return

    s192_pts = []
    for r in results:
        if "s192_r6" in r["name"] and "OutMed" in r["name"]:
            mult = float(r["name"].split("OutMed")[1].split("_")[0])
            s192_pts.append((mult, r["perplexity"], r.get("observed_outlier_rate", 0) * 100))
    if not s192_pts:
        print("no s192_r6 outlier data; skipping fig3")
        return
    s192_pts.sort()
    mults, ppls, outl_pcts = zip(*s192_pts)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    draw_foreground(
        axes[0], mults, ppls,
        color=OURS, label="HQMQ s192_r6 + Med", marker="o",
        is_ours=True, smooth=True, log_x=False,
    )
    draw_reference_hline(axes[0], fp16_ppl, label=f"fp16 ({fp16_ppl:.2f})")
    axes[0].set_xlabel("Outlier multiplier C  (chunks with norm > C x median kept at fp16)")
    axes[0].set_ylabel("Qwen2.5-7B WikiText ppl")
    axes[0].set_yscale("log")
    axes[0].set_title("Outlier threshold vs quality (HQMQ s192_r6)")
    style_axes(axes[0])
    clean_legend(axes[0], loc="upper left")

    # Right panel is a diagnostic, not the OURS curve — use a cool color
    # so amber stays reserved for the headline ppl curve on the left.
    draw_foreground(
        axes[1], mults, outl_pcts,
        color=PALETTE["deep_blue"], marker="s",
        smooth=True, log_x=False,
    )
    axes[1].set_xlabel("Outlier multiplier C")
    axes[1].set_ylabel("Observed outlier fraction (%)")
    axes[1].set_title("Empirical outlier fraction vs threshold")
    style_axes(axes[1])

    fig.tight_layout()
    save_dual(fig, out_dir / "fig3_outlier_sweep")


# ---------------------------------------------------------------------------
# Figure 4 — Memory savings
# ---------------------------------------------------------------------------

def fig4_memory(out_dir):
    """Single-panel KV-cache memory plot.

    Linestyle encodes model (Llama-3-8B = dotted, Llama-3-70B = solid).
    Color encodes config (deep_blue=fp16, cyan/teal/amber for HQMQ tiers).

    Two compact legends: the first names the configs (color), the second
    names the models (linestyle) — both legends draw the *actual* dash pattern
    in their swatch so the dotted / solid distinction is unambiguous.
    """
    from matplotlib.lines import Line2D

    runs_root = Path(__file__).resolve().parents[1] / "runs"
    path = runs_root / "18_memory_accounting.json"
    if not path.exists():
        print("no memory accounting data; skipping fig4")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    table = data["table"]

    configs = [
        ("fp16",         PALETTE["deep_blue"], "fp16 baseline", "o", 5),
        ("hqmq_s24_r3",  PALETTE["cyan"],      "HQMQ s24_r3 (3.17 b/elem)", "o", 5),
        ("hqmq_s96_r4",  PALETTE["teal"],      "HQMQ s96_r4 (3.92 b/elem)", "o", 5),
        ("hqmq_s192_r6", OURS,                 "HQMQ s192_r6 (4.67 b/elem)", "o", 5),
    ]

    model_styles = {
        "Llama-3-8B":  {"linestyle": (0, (2, 2)), "linewidth": 2.0},    # dotted
        "Llama-3-70B": {"linestyle": "-",         "linewidth": 2.4},    # solid
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for model, mstyle in model_styles.items():
        rows = sorted((r for r in table if r["model"] == model),
                      key=lambda r: r["seq_len"])
        if not rows:
            continue
        xs = [r["seq_len"] for r in rows]

        for cfg_key, color, _cfg_label, marker, ms in configs:
            ys_key = "fp16_MB" if cfg_key == "fp16" else f"{cfg_key}_MB"
            ys = [r[ys_key] for r in rows]
            # "x" markers don't have a fill; use the line color for the stroke
            # with a slightly thicker edge so they read against the line.
            if marker == "x":
                mfc, mec, mew = color, color, 1.8
            else:
                mfc, mec, mew = color, "white", 0.7
            ax.plot(
                xs, ys,
                linestyle=mstyle["linestyle"], linewidth=mstyle["linewidth"],
                color=color, marker=marker, markersize=ms,
                markerfacecolor=mfc, markeredgecolor=mec,
                markeredgewidth=mew,
                zorder=4 if cfg_key == "fp16" else 3,
            )

    ax.set_xlabel("Sequence length")
    ax.set_ylabel("KV cache size (MB)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    apply_k_formatter(ax)
    ax.set_title("KV cache memory: fp16 vs HQMQ (Llama-3-8B and Llama-3-70B)")
    style_axes(ax)

    # First legend: configs (color) — drawn as solid lines for clarity.
    # Use the same marker-style treatment as the chart so the fp16 'x' shows
    # in its line color (not invisible against white).
    config_handles = []
    for _, color, label, marker, ms in configs:
        if marker == "x":
            mfc, mec, mew = color, color, 1.8
        else:
            mfc, mec, mew = color, "white", 0.7
        config_handles.append(Line2D(
            [0], [0], color=color, linestyle="-", linewidth=2.4,
            marker=marker, markersize=ms,
            markerfacecolor=mfc, markeredgecolor=mec,
            markeredgewidth=mew, label=label,
        ))
    config_legend = ax.legend(
        handles=config_handles, loc="upper left",
        fontsize=8.5, frameon=False, labelspacing=0.3, handlelength=2.6,
    )
    ax.add_artist(config_legend)

    # Second legend: model (linestyle) — draws the *actual* dotted/solid pattern
    model_handles = [
        Line2D([0], [0], color=PALETTE["axis_gray"],
               linestyle=s["linestyle"], linewidth=s["linewidth"],
               label=name)
        for name, s in model_styles.items()
    ]
    ax.legend(
        handles=model_handles, loc="lower right",
        fontsize=8.5, frameon=False, labelspacing=0.3, handlelength=3.0,
    )

    fig.tight_layout()
    save_dual(fig, out_dir / "fig4_memory")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: paper/image/).")
    args = p.parse_args()
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(__file__).resolve().parents[1] / "paper" / "image"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1_pareto(out_dir)
    fig2_qwen_downstream(out_dir)
    fig3_outlier_sweep(out_dir)
    fig4_memory(out_dir)
    print(f"\nAll figures written to {out_dir}")


if __name__ == "__main__":
    main()
