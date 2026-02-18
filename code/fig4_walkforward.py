"""
Figure 4: Walk-Forward Cross-Validation Results
Shows temporal validation performance across 8 time windows
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
windows = np.array(list(range(1, 9)))
test_r2 = np.array([0.5719, 0.7064, 0.6544, 0.5105, 0.6409, 0.3809, 0.2256, 0.5317])
mean_r2 = 0.5278
std_r2 = 0.1480
shuffled_r2 = 0.648

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

# Bar chart
colors = ['#2E5090' if r2 >= mean_r2 else '#CC79A7' for r2 in test_r2]
bars = ax.bar(windows, test_r2, color=colors, alpha=0.8, edgecolor='black',
              linewidth=1.2, width=0.7)

# Mean line
ax.axhline(y=mean_r2, color='red', linestyle='-', linewidth=2.5,
           label=f'Mean R² = {mean_r2:.3f}', zorder=5)

# Standard deviation band
ax.axhspan(mean_r2 - std_r2, mean_r2 + std_r2, alpha=0.15, color='red',
           label=f'±1 std ({std_r2:.3f})', zorder=0)

# Shuffled CV comparison
ax.axhline(y=shuffled_r2, color='green', linestyle='--', linewidth=2.5,
           label=f'Shuffled CV = {shuffled_r2:.3f}', alpha=0.7, zorder=5)

# Labels and formatting
ax.set_xlabel('Time Window', fontsize=11, fontweight='bold')
ax.set_ylabel('Test R² Score', fontsize=11, fontweight='bold')
ax.set_title('Walk-Forward Cross-Validation: Temporal Generalization',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xticks(windows)
ax.set_xticklabels([f'W{i}' for i in windows], fontsize=9)
ax.tick_params(axis='y', labelsize=9)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
ax.set_ylim(0, 0.8)

# Legend
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

# Annotation explaining temporal ordering
ax.text(4.5, 0.75, 'Respects temporal causality →', fontsize=9,
        ha='center', style='italic', bbox=dict(boxstyle='round,pad=0.4',
        facecolor='wheat', alpha=0.6))

# Highlight worst window
worst_idx = np.argmin(test_r2)
ax.annotate('Worst\nWindow', xy=(windows[worst_idx], test_r2[worst_idx]),
            xytext=(windows[worst_idx] + 0.5, 0.15),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=8, ha='center')

# Highlight best window
best_idx = np.argmax(test_r2)
ax.annotate('Best\nWindow', xy=(windows[best_idx], test_r2[best_idx]),
            xytext=(windows[best_idx] - 0.5, test_r2[best_idx] + 0.05),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=8, ha='center')

plt.tight_layout()

# Save figure
plt.savefig('../figures/fig4_walkforward.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../figures/fig4_walkforward.png', dpi=300, bbox_inches='tight')
print("✓ Figure 4 saved: fig4_walkforward.pdf/png")
plt.close()
