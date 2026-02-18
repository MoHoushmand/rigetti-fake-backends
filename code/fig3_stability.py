"""
Figure 3: Stability Analysis (Two-Panel)
Shows mean R² and variance across layers to identify instability
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure matplotlib
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.0

# Data
layers = np.array([1, 2, 3, 4, 5, 6, 7])
r2_mean = np.array([-0.0562, 0.2121, 0.4907, 0.4607, 0.5376, -0.0546, -0.2509])
r2_std = np.array([0.0761, 0.1400, 0.0655, 0.0330, 0.0571, 0.6899, 0.7960])

# Create two-panel figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Panel (a): Mean R²
ax1.plot(layers, r2_mean, marker='o', markersize=8, color='#2E5090',
         linewidth=2.5, label='Mean R²')
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax1.axvspan(5.5, 7.5, alpha=0.2, color='red', zorder=0)
ax1.set_xlabel('Number of Layers', fontsize=11, fontweight='bold')
ax1.set_ylabel('Mean R² Score', fontsize=11, fontweight='bold')
ax1.set_title('(a) Mean Performance', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.set_xlim(0.5, 7.5)
ax1.tick_params(labelsize=9)
ax1.text(6.5, 0.45, 'Unstable', fontsize=9, ha='center', color='darkred', fontweight='bold')

# Highlight peak
ax1.plot(5, r2_mean[4], 'o', markersize=12, markerfacecolor='gold',
         markeredgecolor='#2E5090', markeredgewidth=2, zorder=10)

# Panel (b): Standard Deviation
ax2.plot(layers, r2_std, marker='s', markersize=8, color='#D55E00',
         linewidth=2.5, label='Std Dev')
ax2.axhline(y=0.1, color='green', linestyle=':', linewidth=2, alpha=0.6)
ax2.axvspan(5.5, 7.5, alpha=0.2, color='red', zorder=0)
ax2.set_xlabel('Number of Layers', fontsize=11, fontweight='bold')
ax2.set_ylabel('R² Standard Deviation', fontsize=11, fontweight='bold')
ax2.set_title('(b) Variance Explosion', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax2.set_xlim(0.5, 7.5)
ax2.set_ylim(0, 0.85)
ax2.tick_params(labelsize=9)

# Annotate stability threshold
ax2.text(7.3, 0.1, 'Stable\nThreshold', fontsize=8, va='center',
         color='green', fontweight='bold')
ax2.text(6.5, 0.7, 'Variance\nExplosion', fontsize=9, ha='center',
         color='darkred', fontweight='bold')

# Highlight explosion points
ax2.plot(layers[5:], r2_std[5:], 'o', markersize=10, markerfacecolor='red',
         markeredgecolor='darkred', markeredgewidth=2, zorder=10)

plt.tight_layout()

# Save figure
plt.savefig('../figures/fig3_stability.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../figures/fig3_stability.png', dpi=300, bbox_inches='tight')
print("✓ Figure 3 saved: fig3_stability.pdf/png")
plt.close()
