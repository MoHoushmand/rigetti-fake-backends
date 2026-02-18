"""
Figure 6: Cross-Platform Comparison
Compares our results with IBM quantum systems
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
systems = ['IBM 4Q\nHeavy-hex', 'IBM 156Q\nHeavy-hex',
           'This work 9Q\n(walk-forward)', 'This work 9Q\n(shuffled)']
r2_scores = np.array([0.764, 0.723, 0.528, 0.648])
colors = ['#4472C4', '#4472C4', '#70AD47', '#A5D86E']
hatches = ['/', '/', 'x', '']

# Create figure
fig, ax = plt.subplots(figsize=(9, 5.5))

# Bar positions
x_pos = np.arange(len(systems))
bars = ax.bar(x_pos, r2_scores, color=colors, alpha=0.85,
              edgecolor='black', linewidth=1.5, width=0.6)

# Add hatching
for i, bar in enumerate(bars):
    if hatches[i]:
        bar.set_hatch(hatches[i])

# Add value labels on bars
for i, (pos, score) in enumerate(zip(x_pos, r2_scores)):
    ax.text(pos, score + 0.02, f'{score:.3f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold')

# Reference line at 0.7
ax.axhline(y=0.7, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.text(3.3, 0.7, 'R² = 0.7', fontsize=9, color='red', fontweight='bold', va='center')

# Labels and formatting
ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax.set_title('Cross-Platform Comparison: QRC Performance',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(systems, fontsize=9, ha='center')
ax.set_ylim(0, 0.85)
ax.tick_params(axis='y', labelsize=9)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')

# Annotations
# IBM systems
ax.annotate('', xy=(0.5, 0.8), xytext=(-0.5, 0.8),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(0, 0.82, 'IBM Quantum Systems', ha='center', fontsize=9,
        fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
        facecolor='lightblue', alpha=0.7))

# Our work
ax.annotate('', xy=(3.5, 0.75), xytext=(1.5, 0.75),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(2.5, 0.77, 'This Work (9-qubit all-to-all)', ha='center', fontsize=9,
        fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
        facecolor='lightgreen', alpha=0.7))

# Explanation box for CV methods
textstr = 'Walk-forward: Respects temporal causality\nShuffled: Randomized train/test splits'
ax.text(0.98, 0.25, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8,
                  edgecolor='black', linewidth=1.2))

# Add system details
details = [
    '4 qubits\nHeavy-hex',
    '156 qubits\nHeavy-hex',
    '9 qubits\nAll-to-all\nTemporal CV',
    '9 qubits\nAll-to-all\nRandom CV'
]
for i, (pos, detail) in enumerate(zip(x_pos, details)):
    ax.text(pos, 0.05, detail, ha='center', va='bottom', fontsize=7,
            style='italic', color='gray')

plt.tight_layout()

# Save figure
plt.savefig('../figures/fig6_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../figures/fig6_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Figure 6 saved: fig6_comparison.pdf/png")
plt.close()
