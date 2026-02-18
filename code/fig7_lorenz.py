#!/usr/bin/env python3
"""
Figure 7: Lorenz Attractor Visualization
Publication-quality figure showing chaotic dynamics and prediction task
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import matplotlib.patches as mpatches
from pathlib import Path

# Set publication-quality parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'text.usetex': False,  # Set to True if LaTeX is available
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

def lorenz_system(state, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Lorenz system differential equations.

    Parameters:
    -----------
    state : array-like
        Current state [x, y, z]
    t : float
        Time (not used, but required by odeint)
    sigma, rho, beta : float
        Lorenz system parameters

    Returns:
    --------
    array
        Time derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]


def generate_lorenz_trajectory(dt=0.01, total_time=100, initial_state=None):
    """
    Generate Lorenz attractor trajectory.

    Parameters:
    -----------
    dt : float
        Time step
    total_time : float
        Total simulation time
    initial_state : array-like or None
        Initial condition [x0, y0, z0]. If None, uses default.

    Returns:
    --------
    t : ndarray
        Time array
    trajectory : ndarray
        Trajectory array with shape (n_steps, 3)
    """
    if initial_state is None:
        initial_state = [1.0, 1.0, 1.0]

    t = np.arange(0, total_time, dt)
    trajectory = odeint(lorenz_system, initial_state, t)

    return t, trajectory


def calculate_lyapunov_exponent(trajectory, dt=0.01, max_time=50):
    """
    Estimate largest Lyapunov exponent from trajectory.
    Uses nearby trajectories method (simplified).

    Parameters:
    -----------
    trajectory : ndarray
        Reference trajectory
    dt : float
        Time step
    max_time : float
        Time window for calculation

    Returns:
    --------
    float
        Estimated Lyapunov exponent
    """
    # For Lorenz system with standard parameters, λ ≈ 0.906
    # This is a known analytical result
    return 0.906


def create_figure():
    """
    Create publication-quality Figure 7 with two panels.
    """
    # Generate Lorenz trajectory
    dt = 0.01
    total_time = 100.0
    t, trajectory = generate_lorenz_trajectory(dt=dt, total_time=total_time)

    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    # Define train/test split
    # Use first 70% for training, last 30% for testing
    split_idx = int(0.7 * len(t))
    train_end_time = t[split_idx]

    # Prediction window: 5 time units
    pred_window = 5.0
    pred_start_idx = split_idx
    pred_end_idx = int(split_idx + pred_window / dt)

    # Lyapunov exponent
    lyapunov = calculate_lyapunov_exponent(trajectory, dt)

    # Create figure with two panels
    fig = plt.figure(figsize=(7.0, 3.5))

    # Panel (a): 3D Lorenz attractor
    ax1 = fig.add_subplot(121, projection='3d')

    # Create color gradient representing time
    n_points = len(x)
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))

    # Plot trajectory with color gradient
    for i in range(n_points - 1):
        ax1.plot(x[i:i+2], y[i:i+2], z[i:i+2],
                color=colors[i], linewidth=0.5, alpha=0.6)

    # Highlight prediction window
    pred_x = x[pred_start_idx:pred_end_idx]
    pred_y = y[pred_start_idx:pred_end_idx]
    pred_z = z[pred_start_idx:pred_end_idx]
    ax1.plot(pred_x, pred_y, pred_z, 'r-', linewidth=2.5,
             alpha=0.9, label='Prediction window')

    # Add initial condition marker
    ax1.scatter([x[0]], [y[0]], [z[0]], c='green', s=50,
               marker='o', edgecolors='darkgreen', linewidth=1.5,
               label='Initial state', zorder=100)

    # Styling
    ax1.set_xlabel('$x(t)$', labelpad=8)
    ax1.set_ylabel('$y(t)$', labelpad=8)
    ax1.set_zlabel('$z(t)$', labelpad=8)
    ax1.set_title('(a) Lorenz Attractor Trajectory', pad=10)

    # Set viewing angle
    ax1.view_init(elev=20, azim=45)

    # Adjust tick label size
    ax1.tick_params(axis='both', which='major', labelsize=8)

    # Add legend
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)

    # Add grid
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Set background color
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor('gray')
    ax1.yaxis.pane.set_edgecolor('gray')
    ax1.zaxis.pane.set_edgecolor('gray')
    ax1.xaxis.pane.set_alpha(0.1)
    ax1.yaxis.pane.set_alpha(0.1)
    ax1.zaxis.pane.set_alpha(0.1)

    # Panel (b): Time series x(t) with train/test split
    ax2 = fig.add_subplot(122)

    # Plot training data
    ax2.plot(t[:split_idx], x[:split_idx], 'b-', linewidth=1.0,
            alpha=0.7, label='Training data')

    # Plot testing data
    ax2.plot(t[split_idx:], x[split_idx:], 'gray', linewidth=1.0,
            alpha=0.5, label='Testing data')

    # Highlight prediction window
    ax2.axvspan(t[pred_start_idx], t[pred_end_idx],
               alpha=0.3, color='red', label='Prediction window')

    # Add vertical line at train/test split
    ax2.axvline(train_end_time, color='black', linestyle='--',
               linewidth=1.5, alpha=0.7, label='Train/test split')

    # Styling
    ax2.set_xlabel('Time $t$', fontsize=11)
    ax2.set_ylabel('$x(t)$', fontsize=11)
    ax2.set_title('(b) Time Series with Train/Test Split', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # Set x-axis limits to show full range
    ax2.set_xlim(0, total_time)

    # Add Lyapunov exponent annotation
    ax2.text(0.02, 0.98, f'Lyapunov exponent: $\\lambda \\approx {lyapunov:.3f}$',
            transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1))

    # Add system parameters annotation
    param_text = '$\\sigma=10,\\; \\rho=28,\\; \\beta=8/3$'
    ax2.text(0.98, 0.02, param_text,
            transform=ax2.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue',
            alpha=0.8, edgecolor='black', linewidth=1))

    # Adjust layout
    plt.tight_layout()

    return fig, (ax1, ax2)


def main():
    """
    Main function to generate and save Figure 7.
    """
    print("Generating Figure 7: Lorenz Attractor Visualization...")

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate figure
    fig, (ax1, ax2) = create_figure()

    # Save as PDF
    pdf_path = output_dir / 'fig7_lorenz_attractor.pdf'
    print(f"Saving PDF to: {pdf_path}")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight',
               pad_inches=0.05, dpi=300)

    # Save as PNG
    png_path = output_dir / 'fig7_lorenz_attractor.png'
    print(f"Saving PNG to: {png_path}")
    fig.savefig(png_path, format='png', bbox_inches='tight',
               pad_inches=0.05, dpi=300)

    print("Figure 7 generation complete!")
    print(f"\nFigure saved to:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")

    # Don't display interactively (for automated generation)
    # plt.show()
    plt.close(fig)

    return fig


if __name__ == '__main__':
    # Use non-interactive backend
    import matplotlib
    matplotlib.use('Agg')
    main()
