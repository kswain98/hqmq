"""Shared visual style for paper / README figures.

Theme features:
  - Restrained color palette with a single accent (amber) for the proposed method.
  - Hidden top/right spines, dark-gray left/bottom spines, no grid.
  - PCHIP-smoothed curves in log-x space (sparse data → smooth lines).
  - White-edged markers for clarity over the smoothed line.
  - "K"-suffixed tick formatter for context lengths (4K, 16K, 32K, ...).
  - PDF + PNG output side by side.

Usage:
    from _plot_style import (PALETTE, OURS, FP16,
                              style_axes, k_formatter,
                              draw_foreground, draw_faded,
                              save_dual)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    draw_foreground(ax, xs, ys, color=OURS, label="HQMQ + Med3x", is_ours=True)
    style_axes(ax)
    save_dual(fig, out_dir / "fig_name")  # → fig_name.pdf + fig_name.png
"""

from __future__ import annotations
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
# Exactly the 5 theme colors from the reference style. Amber is the single
# accent for the proposed method ("OURS"); everything else is a cool tone so
# the eye snaps to the orange line first. Greys are functional (axes, guides,
# reference lines) only — never used for data series.

PALETTE = {
    "deep_blue":   "#3454D1",   # strong cool — primary baseline (matches `full_kv`)
    "cyan":        "#9FE3E1",   # lightest cool — weak baseline (matches `window_kv`)
    "teal":        "#3CB1A8",   # mid cool — alt baseline (matches `streaming_llm`)
    "med_blue":    "#5A87D8",   # mid-strong cool — competitor (matches `infini`)
    "amber":       "#E8A33D",   # OURS accent — the headline method (matches `tc`)
    # --- functional neutrals (NOT data colors) ---
    "axis_gray":   "#444444",   # axes / ticks
    "ref_gray":    "#777777",   # reference lines (fp16, etc) — readable in legend
    "guide_gray":  "#BBBBBB",   # background guide lines (e.g., trained_ctx marker)
}

OURS = PALETTE["amber"]
FP16 = PALETTE["ref_gray"]     # fp16 reference line — visible but subordinate


# Per-method palette used by the Pareto plot. Update as new baselines are added.
# All colors come from the 5-color theme; gray is reserved for the fp16 line.
METHOD_STYLE = {
    "naive int":           {"color": PALETTE["cyan"],      "marker": "x", "ls": "-",  "lw": 2.0, "ms": 6},
    "sph_jl (TurboQuant)": {"color": PALETTE["med_blue"],  "marker": "s", "ls": "--", "lw": 2.0, "ms": 5},
    "HQMQ (no extract)":   {"color": PALETTE["deep_blue"], "marker": "s", "ls": "-",  "lw": 2.0, "ms": 5},
    "HQMQ + Med3x":        {"color": OURS,                 "marker": "o", "ls": "-",  "lw": 2.6, "ms": 6.5},
}


# ---------------------------------------------------------------------------
# Axes / tick styling
# ---------------------------------------------------------------------------

def style_axes(ax):
    """Hide top/right spines, color the remaining spines/ticks dark-gray, drop grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["axis_gray"])
    ax.spines["bottom"].set_color(PALETTE["axis_gray"])
    ax.tick_params(colors=PALETTE["axis_gray"], which="both")
    ax.grid(False)


def k_formatter(x, _pos=None):
    """Format axis tick labels as `4K`, `16K`, `32K` (for context-length axes)."""
    if x >= 1000:
        return f"{int(x / 1000)}K"
    return f"{int(x)}"


def apply_k_formatter(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))


# ---------------------------------------------------------------------------
# Curve smoothing (PCHIP in log-x space, falls back to raw segments)
# ---------------------------------------------------------------------------

def smooth_curve(xs, ys, n_points: int = 300, log_x: bool = True):
    """Monotone-cubic interpolation. Returns (xs_dense, ys_dense).

    Falls back to the raw xs/ys when there aren't enough points or scipy is
    unavailable (so a plot still renders in a minimal environment).

    Robust to duplicate x values: collapses ties by taking the min y at each
    unique x (the Pareto-optimal choice when the y-axis is "lower is better"
    perplexity / latency / error).
    """
    import numpy as np
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 3:
        return xs, ys
    # PCHIP requires strictly increasing x. Sort first, then collapse any duplicates.
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    unique_x, inverse = np.unique(xs, return_inverse=True)
    if len(unique_x) < len(xs):
        # Take min y per unique x — Pareto-frontier semantics
        unique_y = np.full(len(unique_x), np.inf)
        for i, ux_idx in enumerate(inverse):
            if ys[i] < unique_y[ux_idx]:
                unique_y[ux_idx] = ys[i]
        xs = unique_x
        ys = unique_y
    if len(xs) < 3:
        return xs, ys
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:
        return xs, ys
    if log_x and (xs > 0).all():
        u = np.log(xs)
    else:
        u = xs
    spl = PchipInterpolator(u, ys)
    u_dense = np.linspace(u.min(), u.max(), n_points)
    xs_dense = np.exp(u_dense) if log_x and (xs > 0).all() else u_dense
    return xs_dense, spl(u_dense)


# ---------------------------------------------------------------------------
# Line drawing helpers
# ---------------------------------------------------------------------------

def draw_foreground(ax, xs, ys, *, color, label=None, marker="s", markersize=5,
                    linestyle="-", linewidth=2.0, is_ours=False, smooth=True,
                    log_x=True, zorder=3):
    """Solid foreground curve with white-edged markers.

    When `is_ours=True`, bumps linewidth, marker size, and zorder so the curve
    sits visually on top of the baselines.
    """
    if is_ours:
        linewidth = max(linewidth, 2.6)
        markersize = max(markersize, 6.5)
        marker = "o"
        zorder = max(zorder, 4)

    if smooth and len(xs) >= 3:
        xs_s, ys_s = smooth_curve(xs, ys, log_x=log_x)
        ax.plot(xs_s, ys_s, color=color, linestyle=linestyle,
                linewidth=linewidth, zorder=zorder, label=label)
        ax.plot(xs, ys, color=color, linestyle="None",
                marker=marker, markersize=markersize,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.8, zorder=zorder + 1)
    else:
        ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth,
                marker=marker, markersize=markersize,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.8, zorder=zorder, label=label)


def draw_faded(ax, xs, ys, *, color, linestyle="-", linewidth=2.0,
               alpha=0.22, smooth=True, log_x=True, zorder=2):
    """Faded reference curve (e.g., a secondary dataset overlay)."""
    if smooth and len(xs) >= 3:
        xs_s, ys_s = smooth_curve(xs, ys, log_x=log_x)
        ax.plot(xs_s, ys_s, color=color, linestyle=linestyle,
                linewidth=linewidth * 0.9, alpha=alpha, zorder=zorder)
    else:
        ax.plot(xs, ys, color=color, linestyle=linestyle,
                linewidth=linewidth * 0.9, alpha=alpha, zorder=zorder)


def draw_reference_hline(ax, y, *, label=None):
    """Horizontal reference line (e.g., fp16 perplexity).

    Uses dashes rather than dots so the legend marker reads cleanly at the
    small swatch size matplotlib renders.
    """
    ax.axhline(y, color=FP16, linestyle=(0, (5, 3)), linewidth=1.4,
               alpha=1.0, zorder=1, label=label)


def draw_reference_vline(ax, x, *, label=None):
    """Vertical reference line (e.g., trained context length).

    Background guide — lighter and thinner than the hline reference because
    it usually isn't in the legend.
    """
    ax.axvline(x, color=PALETTE["guide_gray"], linestyle=":", linewidth=1.0,
               alpha=0.9, zorder=1, label=label)


# ---------------------------------------------------------------------------
# Legend + figure rcParams
# ---------------------------------------------------------------------------

def clean_legend(ax, *, loc="best", fontsize=9, **kwargs):
    """Frameless, compact legend matching the rest of the theme."""
    return ax.legend(loc=loc, fontsize=fontsize, frameon=False,
                     labelspacing=0.3, handlelength=2.4, **kwargs)


def use_theme_rcparams():
    """Apply matplotlib rcParams that match the theme. Idempotent; call once
    at module import time in plot scripts."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "axes.labelcolor": PALETTE["axis_gray"],
        "xtick.color": PALETTE["axis_gray"],
        "ytick.color": PALETTE["axis_gray"],
        "axes.edgecolor": PALETTE["axis_gray"],
        "axes.linewidth": 0.8,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,    # embed TrueType (vector-text in PDF for arXiv)
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# Saving (PDF + PNG side by side)
# ---------------------------------------------------------------------------

def save_dual(fig, out_stem, *, dpi: int = 180):
    """Save `out_stem.pdf` and `out_stem.png` side by side."""
    out_stem = str(out_stem)
    if out_stem.lower().endswith((".pdf", ".png")):
        out_stem = out_stem.rsplit(".", 1)[0]
    fig.savefig(out_stem + ".pdf", bbox_inches="tight")
    fig.savefig(out_stem + ".png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_stem}.{{pdf,png}}")
