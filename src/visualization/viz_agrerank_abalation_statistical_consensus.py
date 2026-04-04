import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================================
# Global style
# ============================================================================

sns.set_style("whitegrid")

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',

    # Slightly larger than default, but not huge
    'font.family': 'serif',
    'font.size': 11,          # base
    'axes.titlesize': 13,     # subplot titles
    'axes.labelsize': 12,     # x/y labels
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,

    'axes.linewidth': 1.2,
    'figure.dpi': 300,
})

FIG_WIDTH = 6.0       # good for single-column
BAR_HEIGHT = 3.2
SENS_HEIGHT = 3.0

colors = sns.color_palette("Set2", 5)


def beautify_axes(ax):
    """Journal-style axes: white background, black box, normal text."""
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)

    ax.tick_params(axis='both', colors='black', width=1.0)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')


import os
import numpy as np
import matplotlib.pyplot as plt

# Assumes FIG_WIDTH and beautify_axes(ax) are defined elsewhere in your plotting utils

def generate_consensus_bar_plot():
    """
    Bar chart for relevance rate (%) across sparse–dense agreement buckets.
    - Uses percentages (1.98%, 0.08%, <0.01%, <0.01%).
    - Styles the plot to match other figures in the paper.
    - Saves both PNG and PDF in the 'figures' directory.
    """

    # Buckets and empirical probabilities
    buckets = ["Both high", "Dense only", "Sparse only", "Both low"]
    p_rel = [0.0198, 0.00079, 0.0, 0.0]  # probabilities
    percentages = [v * 100 for v in p_rel]  # convert to % (1.98, 0.079, 0.0, 0.0)

    x = np.arange(len(buckets))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.0))

    # Color palette consistent with other figures
    colors = [
        "#d6604d",   # both-high emphasized (coral/red)
        "#2166ac",   # dense-only (blue)
        "#666666",   # sparse-only (dark gray)
        "#999999"    # both-low (light gray)
    ]

    bars = ax.bar(
        x,
        percentages,
        color=colors,
        edgecolor="black",
        linewidth=0.9,
        zorder=3
    )

    def format_percentage(val: float) -> str:
        """Format percentages; floor zeros as '<0.01%'."""
        if val == 0.0:
            return "<0.01%"
        return f"{val:.2f}%"

    # Add labels above bars
    for i, val in enumerate(percentages):
        label = format_percentage(val)
        # Place zero-bucket labels slightly above baseline so they are visible
        y_text = val + 0.07 if val > 0 else 0.07
        ax.text(
            i,
            y_text,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold"
        )

    # Axis labels
    ax.set_ylabel("Relevance rate (%)", fontsize=11, fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, fontsize=10, rotation=20)

    # Y-axis: 0 to just above the max (≈2.0%)
    ax.set_ylim(0, 2.2)

    # Y ticks every 0.5%
    yticks = [0, 0.5, 1.0, 1.5, 2.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.1f}%" for t in yticks])

    # Gridlines and styling
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    beautify_axes(ax)
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/figure_consensus_probability.png",
                bbox_inches="tight", dpi=300)
    plt.savefig("figures/figure_consensus_probability.pdf",
                bbox_inches="tight")

    print("✓ Saved: figure_consensus_probability.png and .pdf")
    plt.close()




# ============================================================================
# Cumulative Benefit over Fusion Input (grouped bars)
# ============================================================================
def generate_scaling_analysis_dl22_only():
    k_values = ['k=200', 'k=500', 'k=1000']
    x = np.arange(len(k_values))
    
    # From Table 5 - TREC DL 2022
    rrf = [0.344, 0.363, 0.373]
    neural = [0.521, 0.563, 0.594]
    agrerank = [0.550, 0.603, 0.627]
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.5))
    
    # Three lines with distinct styles
    ax.plot(x, rrf, 'o:', color='#666666', linewidth=2.5,
            markersize=9, label='Hybrid Fusion (RRF)', alpha=0.8)
    ax.plot(x, neural, 's--', color='#2166ac', linewidth=2.5, 
            markersize=9, label='Neural (RRF+CE)', alpha=0.9,
            markeredgecolor='black', markeredgewidth=0.5)
    ax.plot(x, agrerank, 'D-', color='#d6604d', linewidth=3,
            markersize=10, label='AgreRank', 
            markeredgecolor='black', markeredgewidth=1)
    
    # Label only final values at k=1000 (no +5.6% box)
    ax.text(2.08, agrerank[2], f'{agrerank[2]:.3f}', fontsize=10, 
            fontweight='bold', color='#d6604d', va='center')
    ax.text(2.08, neural[2], f'{neural[2]:.3f}', fontsize=10, 
            fontweight='bold', color='#2166ac', va='center')
    
    ax.set_xlabel('Retrieval Depth', fontsize=11, fontweight='semibold')
    ax.set_ylabel('nDCG@10 (TREC DL 2022)', fontsize=11, fontweight='semibold')
    ax.set_xticks(x)
    ax.set_xticklabels(k_values, fontsize=10)
    
    # Extend y-axis to 0.70 as requested
    ax.set_ylim(0.32, 0.70)
    ax.set_xlim(-0.15, 2.35)
    
    # Gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Legend - upper left, more compact to avoid overlap
    leg = ax.legend(loc='upper left', fontsize=9, framealpha=0.98,
                   frameon=True, fancybox=False,
                   borderpad=0.3, labelspacing=0.3)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1.0)
    
    beautify_axes(ax)
    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/figure3_scaling_analysis.png", 
                bbox_inches="tight", dpi=300)
    plt.savefig("figures/figure3_scaling_analysis.pdf", 
                bbox_inches="tight")
    
    print("✓ Saved: figure3_scaling_analysis.png and .pdf")
    plt.close()



# ============================================================================
# Sensitivity Analysis (two subplots)
# ============================================================================

def generate_sensitivity_analysis():
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(FIG_WIDTH, SENS_HEIGHT)
    )
    plt.subplots_adjust(wspace=0.3)

    alphas = [0.5, 0.65, 0.8, 1.0]
    scores_alpha = [0.6018, 0.6025, 0.5957, 0.5861]

    depths = [50, 100, 200]
    scores_depth = [0.5966, 0.5989, 0.6025]

    # ---- alpha sensitivity ----
    ax1.plot(
        alphas, scores_alpha,
        marker='D', markersize=6,
        linewidth=2.0,
        color='#4C72B0'
    )
    ax1.plot(
        0.65, 0.6025,
        marker='o', markersize=9,
        markerfacecolor='none',
        markeredgecolor='red',
        markeredgewidth=1.7
    )

    ax1.set_title(r"Expansion Weight ($\alpha$)", pad=10)   # << title spacing added
    ax1.set_xlabel(r"Weight $\alpha$")
    ax1.set_ylabel("nDCG@10")
    ax1.set_xticks(alphas)
    ax1.set_ylim(0.58, 0.61)
    ax1.grid(True, linestyle='--', alpha=0.3)

    beautify_axes(ax1)

    # ---- depth sensitivity ----
    ax2.plot(
        depths, scores_depth,
        marker='s', markersize=6,
        linewidth=2.0,
        color='#66c2a5'
    )
    ax2.plot(
        200, 0.6025,
        marker='o', markersize=9,
        markerfacecolor='none',
        markeredgecolor='red',
        markeredgewidth=1.7
    )

    ax2.set_title("Neural Re-ranking Depth ($M$)", pad=10)   # << spacing added
    ax2.set_xlabel("Number of Candidates ($M$)")
    ax2.set_ylabel("nDCG@10")
    ax2.set_xticks(depths)
    ax2.set_ylim(0.59, 0.61)
    ax2.grid(True, linestyle='--', alpha=0.3)

    beautify_axes(ax2)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(
        "figures/figure_sensitivity_analysis.png",
        bbox_inches="tight",
        dpi=300
    )
    print("✓ Saved: figure_sensitivity_analysis.png")



if __name__ == "__main__":
    generate_scaling_analysis_dl22_only()
    generate_sensitivity_analysis()
    generate_consensus_bar_plot()
