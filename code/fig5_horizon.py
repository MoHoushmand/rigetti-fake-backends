"""
Figure 5: Prediction Horizon Decay
Shows R² degradation as prediction horizon increases (in Lyapunov times)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import make_interp_spline

# Configure matplotlib
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.0

# Data
horizon = np.array([1, 2, 5, 10, 20, 50, 100])
lyapunov_tau = np.array([0.01, 0.02, 0.05, 0.09, 0.18, 0.45, 0.91])
r2 = np.array([0.638, 0.634, 0.613, 0.566, 0.430, -0.036, -0.302])

# Create smooth interpolation for curve
horizon_smooth = np.linspace(horizon.min(), horizon.max(), 300)
lyapunov_smooth = np.interp(horizon_smooth, horizon, lyapunov_tau)
spl = make_interp_spline(lyapunov_tau, r2, k=3)
r2_smooth = spl(lyapunov_smooth)

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot smooth decay curve
ax.plot(lyapunov_smooth, r2_smooth, color='#2E5090', linewidth=2.5,
        alpha=0.7, label='Prediction Performance')

# Plot data points
ax.plot(lyapunov_tau, r2, 'o', markersize=9, color='#2E5090',
        markeredgecolor='black', markeredgewidth=1.2, zorder=10,
        label='Measured Points')

# Zero crossing line
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.6)
ax.text(0.95, 0.02, 'R² = 0', fontsize=9, color='red', fontweight='bold', va='bottom')

# Find and mark zero crossing
zero_crossing_idx = np.where(np.diff(np.sign(r2)))[0]
if len(zero_crossing_idx) > 0:
    idx = zero_crossing_idx[0]
    tau_cross = np.interp(0, [r2[idx+1], r2[idx]], [lyapunov_tau[idx+1], lyapunov_tau[idx]])
    ax.axvline(x=tau_cross, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax.plot(tau_cross, 0, 'o', markersize=10, color='orange',
            markeredgecolor='black', markeredgewidth=1.5, zorder=15)
    ax.annotate(f'Zero crossing\n(τ/τ_L ≈ {tau_cross:.2f})',
                xy=(tau_cross, 0), xytext=(tau_cross + 0.15, -0.15),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=9, ha='left', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Shaded regions
ax.axhspan(0, 0.7, alpha=0.1, color='green', zorder=0)
ax.axhspan(-0.35, 0, alpha=0.1, color='red', zorder=0)
ax.text(0.85, 0.55, 'Predictive\nRegime', fontsize=9, ha='right',
        color='darkgreen', fontweight='bold')
ax.text(0.85, -0.25, 'Loss of\nPredictability', fontsize=9, ha='right',
        color='darkred', fontweight='bold')

# Labels and formatting
ax.set_xlabel('Prediction Horizon (τ/τ_L)', fontsize=11, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax.set_title('Prediction Horizon Decay: Lyapunov Time Scaling',
             fontsize=12, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_xlim(0, 1.0)
ax.set_ylim(-0.35, 0.7)
ax.tick_params(labelsize=9)

# Legend
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

# Secondary x-axis for absolute time steps
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
# Map Lyapunov times back to time steps
ax2.set_xticks([0.01, 0.09, 0.18, 0.45, 0.91])
ax2.set_xticklabels(['1', '10', '20', '50', '100'], fontsize=9)
ax2.set_xlabel('Time Steps', fontsize=10, style='italic', color='gray')

plt.tight_layout()

# Save figure
plt.savefig('../figures/fig5_horizon.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../figures/fig5_horizon.png', dpi=300, bbox_inches='tight')
print("✓ Figure 5 saved: fig5_horizon.pdf/png")
plt.close()
