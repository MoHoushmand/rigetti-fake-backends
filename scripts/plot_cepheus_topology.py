#!/usr/bin/env python3
"""
Generate publication-quality figure of FakeCepheus 36Q multi-chip topology.

Four 3x3 chiplets (NW, NE, SW, SE) with intra-chip and inter-chip connectivity.
Output: PDF suitable for IEEE two-column format.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTFILE = Path(__file__).resolve().parent.parent / "paper" / "figures" / "fig12_cepheus_topology.pdf"

# Chiplet definitions: name -> (row_offset, col_offset, qubit_start)
CHIPLETS = {
    "NW": (1, 0, 0),
    "NE": (1, 1, 9),
    "SW": (0, 0, 18),
    "SE": (0, 1, 27),
}

GRID = 3  # 3x3 per chiplet
GAP = 1.6  # spacing between chiplet grids (in grid-unit multiples)

# Inter-chip edges (qubit_a, qubit_b)
INTER_EDGES = [
    # NW <-> NE (right boundary of NW to left boundary of NE)
    (2, 9), (5, 12), (8, 15),
    # NW <-> SW (bottom boundary of NW to top boundary of SW)
    (6, 18), (7, 19), (8, 20),
    # NE <-> SE (bottom boundary of NE to top boundary of SE)
    (15, 27), (16, 28), (17, 29),
    # SW <-> SE (right boundary of SW to left boundary of SE)
    (20, 27), (23, 30), (26, 33),
]

# ---------------------------------------------------------------------------
# Compute qubit positions
# ---------------------------------------------------------------------------
def qubit_positions():
    """Return dict mapping qubit_id -> (x, y)."""
    pos = {}
    for name, (row_off, col_off, q0) in CHIPLETS.items():
        # origin of this chiplet in the global frame
        ox = col_off * (GRID + GAP)
        oy = row_off * (GRID + GAP)
        for idx in range(GRID * GRID):
            r, c = divmod(idx, GRID)
            # row 0 at top within chiplet  ->  y decreases with row
            x = ox + c
            y = oy + (GRID - 1 - r)
            pos[q0 + idx] = (x, y)
    return pos


def intra_edges_for_chiplet(q0):
    """Nearest-neighbour edges within a single 3x3 grid (12 edges)."""
    edges = []
    for idx in range(GRID * GRID):
        r, c = divmod(idx, GRID)
        # right neighbour
        if c + 1 < GRID:
            edges.append((q0 + idx, q0 + idx + 1))
        # down neighbour
        if r + 1 < GRID:
            edges.append((q0 + idx, q0 + idx + GRID))
    return edges


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def main():
    pos = qubit_positions()

    # Gather all intra-chip edges
    intra_edges = []
    for _, (_, _, q0) in CHIPLETS.items():
        intra_edges.extend(intra_edges_for_chiplet(q0))

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Draw chiplet bounding boxes ---
    for name, (row_off, col_off, _) in CHIPLETS.items():
        ox = col_off * (GRID + GAP)
        oy = row_off * (GRID + GAP)
        pad = 0.42
        rect = mpatches.FancyBboxPatch(
            (ox - pad, oy - pad),
            GRID - 1 + 2 * pad,
            GRID - 1 + 2 * pad,
            boxstyle="round,pad=0.15",
            linewidth=1.0,
            edgecolor="#888888",
            facecolor="#F5F5F5",
            zorder=0,
        )
        ax.add_patch(rect)
        # Chiplet label
        ax.text(
            ox + (GRID - 1) / 2,
            oy + GRID - 1 + 0.72,
            name,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#333333",
        )

    # --- Draw intra-chip edges ---
    for a, b in intra_edges:
        xa, ya = pos[a]
        xb, yb = pos[b]
        ax.plot(
            [xa, xb], [ya, yb],
            color="#3B7DD8",
            linewidth=1.4,
            solid_capstyle="round",
            zorder=1,
        )

    # --- Draw inter-chip edges ---
    for a, b in INTER_EDGES:
        xa, ya = pos[a]
        xb, yb = pos[b]
        ax.plot(
            [xa, xb], [ya, yb],
            color="#D63B3B",
            linewidth=1.4,
            linestyle=(0, (4, 3)),
            solid_capstyle="round",
            zorder=2,
        )

    # --- Draw qubit nodes ---
    xs = [pos[q][0] for q in sorted(pos)]
    ys = [pos[q][1] for q in sorted(pos)]
    ax.scatter(
        xs, ys,
        s=220,
        c="white",
        edgecolors="#222222",
        linewidths=1.1,
        zorder=3,
    )

    # --- Qubit labels ---
    for q in sorted(pos):
        x, y = pos[q]
        ax.text(
            x, y,
            str(q),
            ha="center",
            va="center",
            fontsize=5.5,
            fontweight="medium",
            color="#222222",
            zorder=4,
        )

    # --- Legend ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#3B7DD8", linewidth=1.6, label=r"Intra-chip ($F_{2Q}=99.5\%$)"),
        Line2D([0], [0], color="#D63B3B", linewidth=1.6, linestyle=(0, (4, 3)),
               label=r"Inter-chip ($F_{2Q}=99.0\%$)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=True,
        edgecolor="#CCCCCC",
        fontsize=8,
        handlelength=2.2,
    )

    # --- Final layout ---
    ax.margins(0.12)
    fig.tight_layout(pad=0.6)

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTFILE), format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {OUTFILE}")
    print(f"  Intra-chip edges: {len(intra_edges)}  (expected 48)")
    print(f"  Inter-chip edges: {len(INTER_EDGES)}  (expected 12)")
    print(f"  Total qubits:     {len(pos)}  (expected 36)")


if __name__ == "__main__":
    main()
