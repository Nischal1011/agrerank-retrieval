import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.linewidth': 1.2,
    'figure.dpi': 300,
})

FIG_WIDTH = 6.0

def beautify_axes(ax):
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)
    ax.tick_params(axis='both', colors='black', width=1.0)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.minorticks_off()   # <- important

def generate_per_query_robustness_hist(improvements, save_path):
    improvements = np.asarray(improvements, dtype=float)

    median = np.median(improvements)
    mean = np.mean(improvements)
    n = len(improvements)
    n_neg = np.sum(improvements < 0)
    frac_neg = 100.0 * n_neg / n if n > 0 else 0.0
    n_high = np.sum(improvements > 50.0)
    frac_high = 100.0 * n_high / n if n > 0 else 0.0

    # bins go up to 210, last bin is 200+
    bins = list(range(-50, 201, 10)) + [210]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.8))
    counts, edges, patches = ax.hist(
        improvements,
        bins=bins,
        edgecolor='black',
        linewidth=1.0
    )

    for i, patch in enumerate(patches):
        left, right = edges[i], edges[i + 1]
        center = 0.5 * (left + right)
        if center < 0:
            patch.set_facecolor("#f4a6a6")   # negative
        elif center >= 200:
            patch.set_facecolor("#8dd3c7")   # 200+
        else:
            patch.set_facecolor("#8da0cb")   # normal
        patch.set_alpha(0.9)

    ax.axvline(median, color='brown', linestyle='--', linewidth=2)

    ax.set_xlabel("Relative Improvement (%)")
    ax.set_ylabel("Number of Queries")

    # *** NO tick at 200; last tick is at 210 labeled "200+" ***
    ax.set_xlim(-50, 210)
    ax.set_xticks([-50, 0, 50, 100, 150, 210])
    ax.set_xticklabels(["-50", "0", "50", "100", "150", "200+"])

    ymax = max(counts) if len(counts) else 1
    ax.set_ylim(0, ymax + 1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    beautify_axes(ax)

    textstr = (
        f"True Median: {median:.1f}%\n"
        f"True Mean: {mean:.1f}%\n"
        f"Negative Δ: {n_neg}/{n} ({frac_neg:.1f}%)\n"
        f"High Gain (>50%): {n_high}/{n} ({frac_high:.1f}%)"
    )
    ax.text(
        0.98, 0.98, textstr,
        transform=ax.transAxes,
        fontsize=10,
        va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.0)
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"✓ Saved: {save_path}")

def load_improvements_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    per_query = data.get("per_query", [])
    return np.array([d["relative_improvement_pct"] for d in per_query], dtype=float)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--save_path", default="figures/figure_per_query_robustness_journal.png")
    args = parser.parse_args()

    imps = load_improvements_from_json(args.json_path)
    if imps.size == 0:
        print("No per-query data found in JSON.")
    else:
        generate_per_query_robustness_hist(imps, args.save_path)


# python vectorrag/rerank/per_query_analysis_plot.py \
#   --json_path cache/per_query_deltas_msmarco-passage-v2_trec-dl-2022_judged.json \
#   --save_path figures/figure_per_query_msmarco_v2_trecdl2022_k500.png
