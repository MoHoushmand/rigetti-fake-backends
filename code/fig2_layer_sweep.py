"""
Figure 2: Layer Sweep Results - HERO FIGURE
Publication-quality figure showing R² performance and fidelity across circuit depths
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure matplotlib for publication quality
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['patch.linewidth'] = 1.0

# Data
layers = np.array([1, 2, 3, 4, 5, 6, 7])
r2_mean = np.array([-0.0562, 0.2121, 0.4907, 0.4607, 0.5376, -0.0546, -0.2509])
r2_std = np.array([0.0761, 0.1400, 0.0655, 0.0330, 0.0571, 0.6899, 0.7960])
fidelity = np.array([93.0, 86.6, 80.5, 74.9, 69.7, 64.8, 60.3])

# Create figure with two y-axes (IEEE two-column format)
fig, ax1 = plt.subplots(figsize=(7, 4.5))

# R² axis (left)
color_r2 = '#2E5090'  # Professional blue
ax1.set_xlabel('Number of Layers', fontsize=11, fontweight='bold')
ax1.set_ylabel('R² Score', color=color_r2, fontsize=11, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_r2, labelsize=9)
ax1.tick_params(axis='x', labelsize=9)
ax1.set_xlim(0.5, 7.5)
ax1.set_ylim(-0.4, 0.7)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Plot R² with error bars
line1 = ax1.errorbar(layers, r2_mean, yerr=r2_std,
                     marker='o', markersize=8, capsize=5, capthick=1.5,
                     color=color_r2, label='R² Score ± std',
                     linewidth=2, elinewidth=1.5)

# Highlight optimal point (layer 5)
ax1.plot(5, r2_mean[4], 'o', markersize=12, markerfacecolor='gold',
         markeredgecolor=color_r2, markeredgewidth=2, zorder=10)

# Add annotation for optimal point
ax1.annotate('Optimal\n(L=5)', xy=(5, r2_mean[4]), xytext=(4.2, 0.65),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=9, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='black'))

# Fidelity axis (right)
ax2 = ax1.twinx()
color_fidelity = '#D55E00'  # Professional orange
ax2.set_ylabel('Cumulative Fidelity (%)', color=color_fidelity, fontsize=11, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_fidelity, labelsize=9)
ax2.set_ylim(50, 100)

# Plot fidelity
line2 = ax2.plot(layers, fidelity, marker='s', markersize=7,
                color=color_fidelity, label='Cumulative Fidelity',
                linewidth=2, linestyle='--')

# 70% fidelity threshold
ax2.axhline(y=70, color=color_fidelity, linestyle=':', linewidth=2, alpha=0.6)
ax2.text(7.3, 70, '70%', color=color_fidelity, fontsize=8, va='center', fontweight='bold')

# Unstable regime (layers 6-7)
ax1.axvspan(5.5, 7.5, alpha=0.2, color='red', zorder=0)
ax1.text(6.5, 0.6, 'Unstable\nRegime', fontsize=9, fontweight='bold',
         ha='center', va='center', color='darkred',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))

# Phase transition line
ax1.axvline(x=5.5, color='red', linestyle='-.', linewidth=2, alpha=0.7)
ax1.text(5.5, -0.35, 'Phase\nTransition', fontsize=8, ha='center',
         color='red', fontweight='bold')

# Legend
lines = [line1] + line2
labels = ['R² Score ± std', 'Cumulative Fidelity']
ax1.legend(lines, labels, loc='upper left', fontsize=9, framealpha=0.9)

# Title
plt.title('Layer Sweep Analysis: Performance vs. Circuit Depth',
          fontsize=12, fontweight='bold', pad=15)

plt.tight_layout()

# Save figure
plt.savefig('../figures/fig2_layer_sweep.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../figures/fig2_layer_sweep.png', dpi=300, bbox_inches='tight')
print("✓ Figure 2 saved: fig2_layer_sweep.pdf/png")
plt.close()
