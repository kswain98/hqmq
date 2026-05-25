"""Generate the HQMQ method diagram (Figure 0).

Three-panel schematic:
  (a) Quaternion chunking: K[h, t] (d_h=128) -> n_chunks=32 quaternions on S^3.
  (b) Multiplicative composition: q_p in 2T (24-cell vertex) * q_s (random unit).
  (c) Effective codebook: 24 * S codewords from S stored parameters.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def stereographic_project_24cell():
    """Generate stereographic projection of the 24 Hurwitz quaternions to 3D.

    Use a simplified 2D layout: project to the (x, y) plane.
    """
    pts = []
    # 8 axis elements
    for i, (w, x, y, z) in enumerate([
        (1, 0, 0, 0), (-1, 0, 0, 0), (0, 1, 0, 0), (0, -1, 0, 0),
        (0, 0, 1, 0), (0, 0, -1, 0), (0, 0, 0, 1), (0, 0, 0, -1),
    ]):
        pts.append((w, x, y, z))
    # 16 half-integer elements
    for sw in [1, -1]:
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    pts.append((sw * 0.5, sx * 0.5, sy * 0.5, sz * 0.5))
    return np.array(pts)


def make_method_fig(out_path):
    fig = plt.figure(figsize=(13, 4.2))

    # ---- Panel (a): Chunking ----
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("(a) Chunk K/V into quaternions", fontsize=11)

    # Draw K vector as a long horizontal strip
    for i in range(8):
        rect = mpatches.Rectangle((1 + i*0.9, 7), 0.85, 0.7,
                                   facecolor="lightsteelblue", edgecolor="black", lw=0.8)
        ax1.add_patch(rect)
    ax1.text(4.7, 8.1, "$K_t \\in \\mathbb{R}^{d_h}$  (per token, per head)",
             ha="center", fontsize=10)
    ax1.text(4.7, 6.5, "$\\downarrow$ chunk into groups of 4", ha="center", fontsize=9)

    # Draw chunks as quaternion 4-tuples
    chunk_colors = ["#FFB6C1", "#98FB98", "#87CEEB", "#FFD700"]
    for i in range(4):
        for j in range(4):
            rect = mpatches.Rectangle((1.2 + i*2.0 + j*0.35, 4.5), 0.32, 0.7,
                                       facecolor=chunk_colors[i], edgecolor="black", lw=0.7)
            ax1.add_patch(rect)
        ax1.text(1.85 + i*2.0, 3.9, f"$q_{i+1} \\in \\mathbb{{H}}$", ha="center", fontsize=9)

    ax1.text(4.7, 3.0, "Each chunk = one quaternion on $S^3$", ha="center", fontsize=9,
             style="italic")
    ax1.text(4.7, 2.2, "(after radius normalization)", ha="center", fontsize=8, color="gray")

    # ---- Panel (b): Multiplicative composition ----
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.8, 1.8)
    ax2.set_ylim(-1.8, 1.8)
    ax2.set_title("(b) Codebook: $q_p \\cdot q_s$", fontsize=11)

    # Draw $S^3$ as a circle (2D projection)
    circle = mpatches.Circle((0, 0), 1.0, fill=False, edgecolor="gray", lw=1.2, linestyle=":")
    ax2.add_patch(circle)
    ax2.text(0, 1.4, "$S^3$", ha="center", fontsize=11, fontstyle="italic", color="gray")

    # Project 24-cell to 2D (simple projection): use first 2 coordinates
    pts = stereographic_project_24cell()
    rng = np.random.default_rng(0)

    # Plot the 24 primary points
    primary_2d = pts[:, :2]  # take x,y components after w
    for p in pts:
        ax2.plot(p[1], p[2], "o", color="tab:blue", markersize=6, alpha=0.85)
    ax2.text(0, -1.6, "24 primary $q_p \\in 2T$ (Hurwitz)",
             ha="center", fontsize=9, color="tab:blue")

    # Plot S coset rotations: for each q_s, plot 24 rotated points
    n_secondary_to_show = 3
    coset_colors = ["tab:green", "tab:orange", "tab:purple"]
    for k in range(n_secondary_to_show):
        # Random rotation matrix (small perturbation for visualization)
        theta = (k + 1) * 0.6 + 0.3
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        offs = (rng.standard_normal(2) * 0.15)
        for p in pts:
            # Pick (x,y) and rotate
            pp = R @ np.array([p[1], p[2]]) + offs
            # Renormalize to unit
            pp = pp / max(np.linalg.norm(pp), 1e-8)
            ax2.plot(pp[0], pp[1], "o", color=coset_colors[k], markersize=4, alpha=0.6)
    ax2.text(1.55, -1.5, f"+ $S{{=}}3$ shown\n(of $S$ random $q_s$)",
             ha="right", fontsize=8, color="dimgray")
    ax2.text(0, 1.65, "$\\mathcal{C} = \\{q_p \\cdot q_s\\}$  ($24S$ codewords)",
             ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_visible(False)

    # ---- Panel (c): Storage savings ----
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis("off")
    ax3.set_title("(c) $24S$ codewords from $S$ stored", fontsize=11)

    # Comparison: flat learned VQ vs HQMQ
    # Flat VQ: K codewords stored = K
    ax3.text(2.5, 9.2, "Flat learned VQ", ha="center", fontsize=10, fontweight="bold")
    ax3.text(7.5, 9.2, "HQMQ (ours)", ha="center", fontsize=10, fontweight="bold",
             color="tab:blue")

    # Flat: 4608 codewords stored as 4608 quaternions
    n_show = 24
    rect_w, rect_h = 0.15, 0.15
    for i in range(n_show):
        row, col = divmod(i, 8)
        rect = mpatches.Rectangle((0.8 + col*rect_w*2.0, 7.2 - row*rect_h*2.5),
                                   rect_w, rect_h, facecolor="lightgray", edgecolor="black", lw=0.4)
        ax3.add_patch(rect)
    ax3.text(2.5, 5.4, "...", ha="center", fontsize=14, fontweight="bold")
    ax3.text(2.5, 4.7, "$24S$ codewords\n$24S$ parameters\nstored",
             ha="center", fontsize=9)
    ax3.text(2.5, 3.0, "(e.g.\\ 4608 quaternions\nfor $S{=}192$)",
             ha="center", fontsize=8, color="dimgray")

    # HQMQ: 24 primary (free, fixed) + S secondary stored
    # Show 24 primary as outlined boxes
    for i in range(24):
        row, col = divmod(i, 8)
        rect = mpatches.Rectangle((5.4 + col*rect_w*2.0, 7.5 - row*rect_h*2.5),
                                   rect_w, rect_h, facecolor="lightblue",
                                   edgecolor="tab:blue", lw=0.6)
        ax3.add_patch(rect)
    ax3.text(7.2, 5.8, "24 fixed (2T)", ha="center", fontsize=8, color="tab:blue")
    # Show S secondary
    for i in range(3):
        rect = mpatches.Rectangle((6.5 + i*0.4, 4.6), 0.3, 0.3,
                                   facecolor="lightgreen", edgecolor="tab:green", lw=0.7)
        ax3.add_patch(rect)
    ax3.text(7.7, 4.8, "...", fontsize=12)
    ax3.text(7.0, 4.0, "$S$ random $q_s$ stored", ha="center", fontsize=9, color="tab:green")
    ax3.text(7.0, 2.8, "→ $24S$ effective codewords\nfrom $S$ parameters (24$\\times$ savings)",
             ha="center", fontsize=9, color="black", fontweight="bold")
    ax3.text(7.0, 1.4, "no calibration / training\nneeded (Prop.~4.1)",
             ha="center", fontsize=8, color="dimgray", style="italic")

    fig.suptitle("Hurwitz Quaternion Multiplicative Quantization (HQMQ) for KV cache",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parents[1] / "paper" / "image" / "fig0_method.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_method_fig(out_path)
