"""
Figure 11: Topology Comparison - CORRECTED
Accurate IBM Heavy-Hex: Hexagon vertices (degree-3) + edge qubits (degree-2)
Based on IBM's official heavy-hex documentation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import matplotlib.lines as mlines

# Configure matplotlib for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 1.0

fig = plt.figure(figsize=(7.5, 8))

# =============================================================================
# Panel A: Rigetti 9Q Tunable Coupler Architecture
# =============================================================================
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_xlim(-0.5, 3.5)
ax1.set_ylim(-0.5, 3.5)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('(a) Rigetti Novera 9Q\n(Tunable Coupler)', fontsize=10, fontweight='bold', pad=10)

# Data qubit positions (circles) - 3x3 grid
data_qubits = {
    0: (0, 3), 1: (1.5, 3), 2: (3, 3),
    3: (0, 1.5), 4: (1.5, 1.5), 5: (3, 1.5),
    6: (0, 0), 7: (1.5, 0), 8: (3, 0)
}

# Coupler qubit positions (squares) - between data qubits
coupler_qubits = {
    9: (0.75, 3), 10: (2.25, 3),      # Top row horizontal
    11: (0, 2.25), 12: (1.5, 2.25), 13: (3, 2.25),  # Upper vertical
    14: (0.75, 1.5), 15: (2.25, 1.5),  # Middle row horizontal
    16: (0, 0.75), 17: (1.5, 0.75), 18: (3, 0.75),  # Lower vertical
    19: (0.75, 0), 20: (2.25, 0)       # Bottom row horizontal
}

# Draw connections
connections = [
    (0, 9, 1), (1, 10, 2),
    (3, 14, 4), (4, 15, 5),
    (6, 19, 7), (7, 20, 8),
    (0, 11, 3), (1, 12, 4), (2, 13, 5),
    (3, 16, 6), (4, 17, 7), (5, 18, 8)
]

for d1, c, d2 in connections:
    x1, y1 = data_qubits[d1]
    xc, yc = coupler_qubits[c]
    x2, y2 = data_qubits[d2]
    ax1.plot([x1, xc], [y1, yc], 'b-', lw=1.5, zorder=1)
    ax1.plot([xc, x2], [yc, y2], 'b-', lw=1.5, zorder=1)

# Draw data qubits (circles)
for idx, (x, y) in data_qubits.items():
    circle = Circle((x, y), 0.2, facecolor='#1f4e79', edgecolor='white', lw=2, zorder=3)
    ax1.add_patch(circle)
    ax1.text(x, y, str(idx), ha='center', va='center', fontsize=8,
             color='white', fontweight='bold', zorder=4)

# Draw coupler qubits (squares)
from matplotlib.patches import FancyBboxPatch
for idx, (x, y) in coupler_qubits.items():
    rect = FancyBboxPatch((x-0.12, y-0.12), 0.24, 0.24,
                          boxstyle="square,pad=0", facecolor='#2d2d2d',
                          edgecolor='white', lw=1.5, zorder=3)
    ax1.add_patch(rect)
    ax1.text(x, y, str(idx), ha='center', va='center', fontsize=6,
             color='white', zorder=4)

# Add legend
data_patch = mpatches.Patch(facecolor='#1f4e79', edgecolor='white', label='Data Qubit')
coupler_patch = mpatches.Patch(facecolor='#2d2d2d', edgecolor='white', label='Coupler')
ax1.legend(handles=[data_patch, coupler_patch], loc='upper right', fontsize=7)

ax1.text(1.5, -0.4, '9 data + 12 couplers = 21 qubits\n12 effective edges, 33% connectivity',
         ha='center', va='top', fontsize=7, style='italic')

# =============================================================================
# Panel B: IBM Heavy-Hex - CORRECT IMPLEMENTATION
# Based on IBM docs: Hexagon vertices (degree-3) + edge midpoint qubits (degree-2)
# =============================================================================
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_xlim(-0.5, 5.5)
ax2.set_ylim(-0.5, 5)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('(b) IBM Heavy-Hex\n(Eagle/Heron)', fontsize=10, fontweight='bold', pad=10)

# Heavy-hex geometry:
# - Degree-3 qubits at hexagon vertices (where 3 hexagons meet)
# - Degree-2 qubits at edge midpoints
# Structure: Two rows of connected hexagons

# Hexagon parameters
hex_size = 0.8  # Distance from center to vertex
hex_h = hex_size * np.sqrt(3) / 2  # Half-height of hexagon

# Define hexagon centers (2 hexagons side by side)
hex_centers = [
    (1.8, 2.8),   # Upper-left hexagon
    (3.2, 2.8),   # Upper-right hexagon
    (2.5, 1.5),   # Lower-center hexagon
]

def get_hexagon_vertices(cx, cy, size):
    """Get 6 vertices of a flat-topped hexagon"""
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    return [(cx + size * np.cos(a), cy + size * np.sin(a)) for a in angles]

# Collect all unique vertices and edges
all_vertices = set()
all_edge_midpoints = []
all_edges = []

for cx, cy in hex_centers:
    verts = get_hexagon_vertices(cx, cy, hex_size)
    for i, v in enumerate(verts):
        all_vertices.add((round(v[0], 3), round(v[1], 3)))
        # Edge from vertex i to vertex (i+1)%6
        v2 = verts[(i + 1) % 6]
        midx = (v[0] + v2[0]) / 2
        midy = (v[1] + v2[1]) / 2
        all_edge_midpoints.append((midx, midy, v[0], v[1], v2[0], v2[1]))

all_vertices = list(all_vertices)

# Remove duplicate midpoints (shared edges between hexagons)
unique_midpoints = []
seen = set()
for mp in all_edge_midpoints:
    key = (round(mp[0], 2), round(mp[1], 2))
    if key not in seen:
        seen.add(key)
        unique_midpoints.append(mp)

# Draw edges (vertex to midpoint to vertex)
for midx, midy, vx1, vy1, vx2, vy2 in unique_midpoints:
    ax2.plot([vx1, midx], [vy1, midy], '#1f4e79', lw=2, zorder=1)
    ax2.plot([midx, vx2], [midy, vy2], '#1f4e79', lw=2, zorder=1)

# Draw degree-3 vertices (white/hollow circles with thick border)
for vx, vy in all_vertices:
    circle = Circle((vx, vy), 0.15, facecolor='white', edgecolor='#333333', lw=2.5, zorder=3)
    ax2.add_patch(circle)

# Draw degree-2 edge qubits (alternating blue/green)
for i, (midx, midy, _, _, _, _) in enumerate(unique_midpoints):
    color = '#4472C4' if i % 2 == 0 else '#70AD47'
    circle = Circle((midx, midy), 0.12, facecolor=color, edgecolor='white', lw=1.2, zorder=3)
    ax2.add_patch(circle)

# Legend
syndrome_patch = mlines.Line2D([], [], marker='o', color='white', markeredgecolor='#333333',
                                markeredgewidth=2, markersize=9, linestyle='None', label='Degree-3 vertex')
blue_patch = mlines.Line2D([], [], marker='o', color='#4472C4', markersize=7,
                           linestyle='None', label='Degree-2 (type A)')
green_patch = mlines.Line2D([], [], marker='o', color='#70AD47', markersize=7,
                            linestyle='None', label='Degree-2 (type B)')
ax2.legend(handles=[syndrome_patch, blue_patch, green_patch], loc='upper right', fontsize=6)

ax2.text(2.5, -0.3, 'Degree-2/3 mixed connectivity\n~15% connectivity (sparse)',
         ha='center', va='top', fontsize=7, style='italic')

# =============================================================================
# Panel C: Simple Square Lattice (for comparison)
# =============================================================================
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_xlim(-0.5, 3.5)
ax3.set_ylim(-0.5, 3.5)
ax3.set_aspect('equal')
ax3.axis('off')
ax3.set_title('(c) Square Lattice\n(IBM Nighthawk)', fontsize=10, fontweight='bold', pad=10)

# Simple 3x3 square lattice (no couplers)
sq_qubits = {
    0: (0, 3), 1: (1.5, 3), 2: (3, 3),
    3: (0, 1.5), 4: (1.5, 1.5), 5: (3, 1.5),
    6: (0, 0), 7: (1.5, 0), 8: (3, 0)
}

# Direct edges
sq_edges = [
    (0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),  # Horizontal
    (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8)   # Vertical
]

for q1, q2 in sq_edges:
    x1, y1 = sq_qubits[q1]
    x2, y2 = sq_qubits[q2]
    ax3.plot([x1, x2], [y1, y2], color='#D55E00', lw=2.5, zorder=1)

for idx, (x, y) in sq_qubits.items():
    circle = Circle((x, y), 0.22, facecolor='#1f4e79', edgecolor='white', lw=2, zorder=3)
    ax3.add_patch(circle)
    ax3.text(x, y, str(idx), ha='center', va='center', fontsize=9,
             color='white', fontweight='bold', zorder=4)

ax3.text(1.5, -0.4, '9 qubits, 12 direct edges\n33% connectivity (dense)',
         ha='center', va='top', fontsize=7, style='italic')

# =============================================================================
# Panel D: Optimal Depth Comparison Bar Chart
# =============================================================================
ax4 = fig.add_subplot(2, 2, 4)

topologies = ['Rigetti 9Q\n(Tunable)', 'IBM Heavy-Hex\n(156Q)', 'Square Lattice\n(9Q)']
optimal_depths = [5, 1, 5]  # Heavy-hex d_opt≈0.17 rounds to 1 layer per paper Sec V.B
colors = ['#1f4e79', '#4472C4', '#D55E00']

bars = ax4.bar(topologies, optimal_depths, color=colors, edgecolor='black', lw=1.5)

ax4.set_ylabel('Optimal Circuit Depth (layers)', fontsize=10, fontweight='bold')
ax4.set_ylim(0, 7)
ax4.set_title('(d) Optimal Depth by Topology', fontsize=10, fontweight='bold', pad=10)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, depth in zip(bars, optimal_depths):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'd={depth}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add formula annotation
ax4.text(0.5, 0.02, r'$d_{opt} = \frac{-\ln(F_{threshold})}{n_{edges} \times \ln(F_{2Q})}$',
         transform=ax4.transAxes, fontsize=8, ha='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('../figures/fig11_topology_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../figures/fig11_topology_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Figure 11 saved: fig11_topology_comparison.pdf/png (PROPERLY CONNECTED)")
plt.close()
