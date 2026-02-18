"""
Figure 1: QRC Architecture - CORRECTED
Shows accurate Rigetti 9Q tunable coupler architecture with 21 total qubits
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# Configure matplotlib for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 1.0

fig = plt.figure(figsize=(7.5, 9))

# =============================================================================
# Panel A: Rigetti 9Q Tunable Coupler Topology (CORRECTED)
# =============================================================================
ax1 = fig.add_subplot(3, 1, 1)
ax1.set_xlim(-0.8, 4.3)
ax1.set_ylim(-0.8, 4.0)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('(a) Rigetti Novera 9Q Tunable Coupler Topology', fontsize=11, fontweight='bold', pad=10)

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
    ax1.plot([x1, xc], [y1, yc], '#D55E00', lw=2, zorder=1)
    ax1.plot([xc, x2], [yc, y2], '#D55E00', lw=2, zorder=1)

# Draw data qubits (circles)
for idx, (x, y) in data_qubits.items():
    circle = Circle((x, y), 0.22, facecolor='#1f4e79', edgecolor='white', lw=2.5, zorder=3)
    ax1.add_patch(circle)
    ax1.text(x, y, str(idx), ha='center', va='center', fontsize=9,
             color='white', fontweight='bold', zorder=4)

# Draw coupler qubits (squares)
for idx, (x, y) in coupler_qubits.items():
    rect = FancyBboxPatch((x-0.14, y-0.14), 0.28, 0.28,
                          boxstyle="square,pad=0", facecolor='#2d2d2d',
                          edgecolor='white', lw=1.5, zorder=3)
    ax1.add_patch(rect)
    ax1.text(x, y, str(idx), ha='center', va='center', fontsize=6,
             color='white', zorder=4)

# Add legend and stats
data_patch = mpatches.Patch(facecolor='#1f4e79', edgecolor='white', label='Data Qubit (9)')
coupler_patch = mpatches.Patch(facecolor='#2d2d2d', edgecolor='white', label='Tunable Coupler (12)')
edge_line = Line2D([0], [0], color='#D55E00', lw=2, label='CZ Gate Connection')
ax1.legend(handles=[data_patch, coupler_patch, edge_line], loc='upper right', fontsize=8)

# Hardware specs annotation
specs_text = ('Hardware Specifications:\n'
              '• 9 data qubits + 12 tunable couplers = 21 total\n'
              '• 12 effective CZ gate edges\n'
              '• $F_{1Q}$ = 99.9%, $F_{2Q}$ = 99.4%\n'
              '• $T_1 = T_2$ = 27 μs\n'
              '• Connectivity: 33.3%')
ax1.text(3.8, 0.5, specs_text, fontsize=8, va='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))

# =============================================================================
# Panel B: QRC Circuit Layer Structure
# =============================================================================
ax2 = fig.add_subplot(3, 1, 2)
ax2.set_xlim(-0.5, 10)
ax2.set_ylim(-0.5, 9.5)
ax2.axis('off')
ax2.set_title('(b) QRC Circuit Structure (Single Layer)', fontsize=11, fontweight='bold', pad=10)

# Draw qubit wires (9 qubits)
wire_y = np.arange(9)[::-1]  # 8,7,6,...,0
for i, y in enumerate(wire_y):
    ax2.plot([0, 9.5], [y, y], 'k-', lw=0.8, zorder=1)
    ax2.text(-0.3, y, f'q{8-i}', ha='right', va='center', fontsize=8, fontweight='bold')

# Input encoding gates (green)
for i, y in enumerate(wire_y):
    rect = FancyBboxPatch((0.5, y-0.25), 0.8, 0.5, boxstyle="round,pad=0.05",
                          facecolor='#70AD47', edgecolor='black', lw=1, zorder=2)
    ax2.add_patch(rect)
    ax2.text(0.9, y, r'$R_Y(\theta)$', ha='center', va='center', fontsize=6, fontweight='bold')

ax2.text(0.9, -0.3, 'Input\nEncoding', ha='center', va='top', fontsize=8, fontweight='bold', color='#70AD47')

# Variational rotation gates (blue)
for i, y in enumerate(wire_y):
    rect = FancyBboxPatch((2, y-0.25), 0.8, 0.5, boxstyle="round,pad=0.05",
                          facecolor='#4472C4', edgecolor='black', lw=1, zorder=2)
    ax2.add_patch(rect)
    ax2.text(2.4, y, r'$R_Y(\phi_i)$', ha='center', va='center', fontsize=6, color='white', fontweight='bold')

ax2.text(2.4, -0.3, 'Trainable\nRotations', ha='center', va='top', fontsize=8, fontweight='bold', color='#4472C4')

# CZ gates (orange) - draw according to actual topology
cz_pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
            (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8)]

# Map qubit index to wire y-position
q_to_y = {i: wire_y[8-i] for i in range(9)}

cz_x = 4.2
for q1, q2 in cz_pairs:
    y1, y2 = q_to_y[q1], q_to_y[q2]
    # Draw CZ gate
    ax2.plot([cz_x, cz_x], [y1, y2], color='#D55E00', lw=2, zorder=2)
    ax2.plot(cz_x, y1, 'o', color='#D55E00', markersize=6, zorder=3)
    ax2.plot(cz_x, y2, 'o', color='#D55E00', markersize=6, zorder=3)
    cz_x += 0.25

ax2.text(5.5, -0.3, 'CZ Entanglement\n(12 gates)', ha='center', va='top', fontsize=8, fontweight='bold', color='#D55E00')

# Measurement (red)
for i, y in enumerate(wire_y):
    rect = FancyBboxPatch((8, y-0.25), 0.8, 0.5, boxstyle="round,pad=0.05",
                          facecolor='#C00000', edgecolor='black', lw=1, zorder=2)
    ax2.add_patch(rect)
    ax2.text(8.4, y, 'M', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

ax2.text(8.4, -0.3, 'Z-basis\nMeasurement', ha='center', va='top', fontsize=8, fontweight='bold', color='#C00000')

# Layer annotation
ax2.annotate('', xy=(7.5, 9.2), xytext=(1.5, 9.2),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax2.text(4.5, 9.5, 'One Layer (repeated $d$ times)', ha='center', va='bottom', fontsize=9, fontweight='bold')

# =============================================================================
# Panel C: Complete QRC Pipeline
# =============================================================================
ax3 = fig.add_subplot(3, 1, 3)
ax3.set_xlim(-0.5, 10.5)
ax3.set_ylim(-0.5, 2.5)
ax3.axis('off')
ax3.set_title('(c) Complete QRC System Pipeline', fontsize=11, fontweight='bold', pad=10)

# Pipeline boxes
boxes = [
    (0, 'Time Series\nInput\n$x(t)$', '#E7E6E6', 'black'),
    (2.2, 'Quantum\nReservoir\n(d layers)', '#1f4e79', 'white'),
    (4.4, 'Feature\nExtraction\n(45 features)', '#70AD47', 'white'),
    (6.6, 'Ridge\nRegression\n$\\alpha=1.0$', '#4472C4', 'white'),
    (8.8, 'Prediction\nOutput\n$\\hat{y}(t+1)$', '#E7E6E6', 'black'),
]

for x, text, color, tcolor in boxes:
    rect = FancyBboxPatch((x, 0.5), 1.8, 1.5, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='black', lw=1.5, zorder=2)
    ax3.add_patch(rect)
    ax3.text(x + 0.9, 1.25, text, ha='center', va='center', fontsize=8,
             color=tcolor, fontweight='bold', linespacing=1.2)

# Arrows
for i in range(4):
    ax3.annotate('', xy=(2.0 + i*2.2, 1.25), xytext=(1.85 + i*2.2, 1.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Feature details
feature_text = ('Features: $\\langle Z_i \\rangle$ (9) + $\\langle Z_i Z_j \\rangle$ (36) = 45 total')
ax3.text(5.25, 0.1, feature_text, ha='center', va='top', fontsize=8, style='italic')

# Performance annotation
perf_text = ('Optimal: $d=5$ layers, $R^2=0.538$, Walk-forward CV: $R^2=0.528 \\pm 0.148$')
ax3.text(5.25, 2.3, perf_text, ha='center', va='bottom', fontsize=8,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('../figures/fig1_architecture.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../figures/fig1_architecture.png', dpi=300, bbox_inches='tight')
print("✓ Figure 1 saved: fig1_architecture.pdf/png (CORRECTED with tunable couplers)")
plt.close()
