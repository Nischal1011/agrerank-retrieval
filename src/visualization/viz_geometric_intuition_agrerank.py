import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.path import Path
import matplotlib.patheffects as path_effects

def plot_geometric_intuition():
    """
    Create a publication-quality geometric intuition figure using modern matplotlib styling.
    """
    
    # Set style for publication quality
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'axes.linewidth': 0,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5
    })
    
    fig = plt.figure(figsize=(14, 10), facecolor='white', dpi=150)
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_facecolor('#f8f9fa')
    ax.grid(False)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Origin
    origin = np.array([0, 0])
    
    # Define vectors with better angles for visualization
    q_orig = np.array([0.70, 0.714])  # ~45 degrees, length 1.0
    q_exp = np.array([0.866, 0.5])    # ~30 degrees, length 1.0
    
    # Generate document clusters with better distribution
    np.random.seed(42)
    n_docs = 25
    
    # Relevant cluster: tightly around q_orig
    angle_orig = np.arctan2(q_orig[1], q_orig[0])
    rel_angles = np.random.normal(angle_orig, 0.15, n_docs)
    rel_radii = np.random.normal(0.68, 0.08, n_docs)
    rel_x = rel_radii * np.cos(rel_angles)
    rel_y = rel_radii * np.sin(rel_angles)
    
    # Drift cluster: orthogonal to q_orig (near horizontal axis)
    drift_angles = np.random.normal(0.1, 0.12, n_docs)
    drift_radii = np.random.normal(0.75, 0.06, n_docs)
    drift_x = drift_radii * np.cos(drift_angles)
    drift_y = drift_radii * np.sin(drift_angles)
    
    # ========== PLOT DOCUMENT CLUSTERS ==========
    
    # Plot drift documents first (background layer)
    scatter_drift_glow = ax.scatter(drift_x, drift_y, 
                                     s=350, c='#ef4444', alpha=0.15,
                                     marker='X', zorder=2)
    scatter_drift = ax.scatter(drift_x, drift_y, 
                                s=180, c='#ef4444', alpha=0.85,
                                marker='X', edgecolors='#991b1b', 
                                linewidths=2.5, zorder=3)
    
    # Plot relevant documents (foreground layer)
    scatter_rel_glow = ax.scatter(rel_x, rel_y, 
                                   s=300, c='#10b981', alpha=0.2,
                                   zorder=4)
    scatter_rel = ax.scatter(rel_x, rel_y, 
                              s=150, c='#10b981', alpha=0.9,
                              edgecolors='#047857', linewidths=2.5, 
                              zorder=5)
    
    # ========== PLOT VECTORS WITH FANCY ARROWS ==========
    
    # Original query vector (solid, thick, black)
    arrow_orig = FancyArrowPatch(
        origin, q_orig * 0.95,
        arrowstyle='->,head_width=0.6,head_length=0.8',
        color='#0f172a',
        linewidth=5,
        zorder=10,
        path_effects=[path_effects.Stroke(linewidth=7, foreground='#cbd5e1', alpha=0.3),
                      path_effects.Normal()]
    )
    ax.add_patch(arrow_orig)
    
    # Expanded query vector (dashed, blue)
    arrow_exp = FancyArrowPatch(
        origin, q_exp * 0.95,
        arrowstyle='->,head_width=0.55,head_length=0.75',
        color='#2563eb',
        linewidth=4.5,
        linestyle='--',
        zorder=9,
        alpha=0.9,
        path_effects=[path_effects.Stroke(linewidth=6.5, foreground='#93c5fd', alpha=0.3),
                      path_effects.Normal()]
    )
    ax.add_patch(arrow_exp)
    
    # ========== ANGLE ARC ==========
    
    angle_orig_deg = np.degrees(np.arctan2(q_orig[1], q_orig[0]))
    angle_exp_deg = np.degrees(np.arctan2(q_exp[1], q_exp[0]))
    
    arc = mpatches.Arc(
        origin, 0.5, 0.5,
        theta1=angle_exp_deg, theta2=angle_orig_deg,
        color='#64748b', linewidth=3, zorder=8
    )
    ax.add_patch(arc)
    
    # Theta label with shadow effect
    theta_text = ax.text(0.28, 0.20, r'$\theta$', 
                         fontsize=24, style='italic', 
                         color='#475569', weight='bold', zorder=11)
    theta_text.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground='white'),
        path_effects.Normal()
    ])
    
    # ========== VECTOR LABELS ==========
    
    # Original query label
    orig_label = ax.text(q_orig[0] + 0.12, q_orig[1] + 0.08,
                         r'$\mathbf{v}_q$' + '\n(Original Query)',
                         fontsize=13, weight='bold', color='#0f172a',
                         ha='left', va='bottom', zorder=11,
                         bbox=dict(boxstyle='round,pad=0.4', 
                                   facecolor='white', edgecolor='#0f172a',
                                   linewidth=2, alpha=0.95))
    
    # Expanded query label
    exp_label = ax.text(q_exp[0] + 0.05, q_exp[1] - 0.12,
                        r"$\mathbf{v}_{q'}$" + '\n(Expanded Query)',
                        fontsize=12, weight='semibold', color='#2563eb',
                        ha='left', va='top', zorder=11,
                        bbox=dict(boxstyle='round,pad=0.4',
                                  facecolor='white', edgecolor='#2563eb',
                                  linewidth=2, alpha=0.95))
    
    # ========== ANNOTATION BOXES ==========
    
    # Green box for relevant documents
    green_box = FancyBboxPatch(
        (0.05, 0.88), 0.28, 0.09,
        boxstyle="round,pad=0.01",
        facecolor='#ecfdf5', edgecolor='#10b981',
        linewidth=3, transform=ax.transAxes, zorder=12
    )
    ax.add_patch(green_box)
    
    ax.text(0.19, 0.945, '✓ High Alignment',
            fontsize=12, weight='bold', color='#047857',
            ha='center', va='center', transform=ax.transAxes)
    ax.text(0.19, 0.905, r'$\cos(\theta) \approx 0.95$',
            fontsize=11, color='#059669',
            ha='center', va='center', transform=ax.transAxes)
    
    # Red box for drift documents
    red_box = FancyBboxPatch(
        (0.67, 0.05), 0.28, 0.09,
        boxstyle="round,pad=0.01",
        facecolor='#fef2f2', edgecolor='#ef4444',
        linewidth=3, transform=ax.transAxes, zorder=12
    )
    ax.add_patch(red_box)
    
    ax.text(0.81, 0.115, '✗ Orthogonal Drift',
            fontsize=12, weight='bold', color='#b91c1c',
            ha='center', va='center', transform=ax.transAxes)
    ax.text(0.81, 0.075, r'$\cos(\theta) \approx 0.10$',
            fontsize=11, color='#dc2626',
            ha='center', va='center', transform=ax.transAxes)
    
    # Yellow insight box at bottom
    insight_box = FancyBboxPatch(
        (0.15, 0.02), 0.70, 0.08,
        boxstyle="round,pad=0.015",
        facecolor='#fffbeb', edgecolor='#f59e0b',
        linewidth=3.5, transform=ax.transAxes, zorder=12
    )
    ax.add_patch(insight_box)
    
    ax.text(0.50, 0.075, 'Multiplicative Gating Formula',
            fontsize=13, weight='bold', color='#92400e',
            ha='center', va='center', transform=ax.transAxes)
    ax.text(0.50, 0.04, r'$S_{\mathrm{final}}(d) = S_{\mathrm{PRF}}(d) \times \max(0, \cos(\theta))$',
            fontsize=12, color='#78350f',
            ha='center', va='center', transform=ax.transAxes)
    
    # ========== ORIGIN MARKER ==========
    origin_circle = Circle(origin, 0.025, color='#475569', zorder=11)
    ax.add_patch(origin_circle)
    
    # ========== FINALIZE PLOT ==========
    
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Title
    plt.title('Multiplicative Drift Gating: Geometric Intuition',
              fontsize=18, weight='bold', pad=25, color='#0f172a')
    
    plt.tight_layout()
    
    # Save with high quality
    output_path = 'figure_geometric_intuition_publication.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.replace('.png', '.pdf'), 
                bbox_inches='tight', facecolor='white')
    
    plt.show()
    print(f"✅ High-quality figure saved to:")
    print(f"   • {output_path}")
    print(f"   • {output_path.replace('.png', '.pdf')}")

if __name__ == "__main__":
    plot_geometric_intuition()