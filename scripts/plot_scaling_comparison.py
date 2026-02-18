#!/usr/bin/env python3
"""
Generate publication-quality figure: Optimal Depth Scaling vs Edge Count.

Shows d_opt = ln(0.70) / (n_edges * ln(F_2Q)) for F_2Q = 0.994,
with annotated processor data points.

Output: PDF suitable for IEEE single-column format (3.5 in wide).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTFILE = (
    Path(__file__).resolve().parent.parent
    / "paper"
    / "figures"
    / "fig13_scaling_comparison.pdf"
)

F_2Q = 0.994
F_THRESHOLD = 0.70


def d_opt(n_edges):
    """Optimal circuit depth for 70% fidelity threshold."""
    return np.log(F_THRESHOLD) / (n_edges * np.log(F_2Q))


def main():
    # IEEE-compatible serif fonts
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "mathtext.fontset": "cm",
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.45,
    })

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # ── Smooth curve ───────────────────────────────────────────────────────
    n_arr = np.linspace(5, 620, 2000)
    d_arr = d_opt(n_arr)

    ax.plot(n_arr, d_arr, "k-", linewidth=1.0, zorder=3,
            label=r"$d_{\mathrm{opt}} = \ln(0.70)\,/\,(n_{\mathrm{edges}}\ln F_{2Q})$")

    # ── Crossover where d_opt = 1 ─────────────────────────────────────────
    n_cross = np.log(F_THRESHOLD) / np.log(F_2Q)  # ~59.3

    # Horizontal dashed line at d_opt = 1
    ax.axhline(y=1.0, color="#777777", linestyle="--", linewidth=0.65, zorder=2)
    # Place "Single-layer limit" as text right on the dashed line, far right
    ax.text(
        610, 1.05, "Single-layer\nlimit",
        fontsize=5.5, va="bottom", ha="right", color="#666666",
        linespacing=0.85,
    )

    # ── Shaded "not physical" region ──────────────────────────────────────
    ax.fill_between(
        n_arr, 0, 1.0,
        where=(n_arr >= n_cross),
        color="#ffe0e0", alpha=0.55, zorder=1, linewidth=0,
    )
    ax.text(
        400, 0.50, "Circuit depth $< 1$\n(not physical)",
        fontsize=6, ha="center", va="center", color="#aa2222",
        fontstyle="italic",
    )

    # ── Processor markers and annotations ─────────────────────────────────
    # Each entry: (label, n_edges, marker, color, ms, text_x, text_y, ha)
    # Positions chosen to avoid all overlaps:
    #   - Novera:  marker at (12, 4.94) -> label upper-right in open space
    #   - Cepheus: marker at (60, 0.99) -> label above, shifted right
    #   - Ankaa-3: marker at (140,0.42) -> label above-right
    #   - Future:  marker at (560,0.11) -> label above-left
    processors = [
        ("Novera\n(9Q)",    12,  "o", "#1f77b4", 5.5,  100,  5.4, "left"),
        ("Cepheus\n(36Q)",  60,  "D", "#d62728", 5.0,  140,  2.6, "left"),
        ("Ankaa-3\n(84Q)",  140, "^", "#2ca02c", 5.5,  270,  1.9, "left"),
        ("Future\n(336Q)",  560, "s", "#7f7f7f", 5.0,  440,  2.4, "left"),
    ]

    for label, n_e, marker, color, ms, tx, ty, ha in processors:
        d_val = d_opt(n_e)
        ax.plot(
            n_e, d_val, marker=marker, color=color,
            markersize=ms, markeredgecolor="black", markeredgewidth=0.5,
            zorder=5, linestyle="none",
        )
        ax.annotate(
            label,
            xy=(n_e, d_val),
            xytext=(tx, ty),
            fontsize=6, color=color, fontweight="bold",
            ha=ha, va="center",
            linespacing=0.85,
            arrowprops=dict(
                arrowstyle="-|>",
                color="#999999",
                linewidth=0.5,
                shrinkA=0,
                shrinkB=2,
                mutation_scale=5,
            ),
            zorder=6,
        )

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_xlabel(r"Edge count $n_{\mathrm{edges}}$")
    ax.set_ylabel(r"$d_{\mathrm{opt}}$")
    ax.set_xlim(0, 620)
    ax.set_ylim(0, 7)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.grid(True, which="major", linestyle="-", linewidth=0.35, alpha=0.40)

    # Legend -- small, upper-right, just the formula
    ax.legend(
        loc="upper right", frameon=True, edgecolor="#cccccc",
        fancybox=False, framealpha=0.93, handlelength=1.5,
        borderpad=0.4,
    )

    fig.tight_layout(pad=0.3)
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTFILE), format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Summary
    print(f"Saved: {OUTFILE}")
    print(f"  F_2Q = {F_2Q},  threshold = {F_THRESHOLD}")
    print(f"  Crossover (d_opt=1) at n_edges = {n_cross:.1f}")
    for _, n_e, *rest in processors:
        label_flat = rest[5 - 3]  # color field
        print(f"  n_edges={n_e:>3d},  d_opt={d_opt(n_e):.3f}")


if __name__ == "__main__":
    main()
