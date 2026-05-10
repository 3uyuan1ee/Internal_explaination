"""
Visualization script for CBM V2 intervention experiment results.

Generates 6 core figures from the saved intervention results:
  Fig 1: Error Attribution Pie Chart
  Fig 3: Intervention Efficiency Curves
  Fig 4: Strategy AUC Bar Chart
  Fig 5: k_min Distribution Histogram
  Fig 7: Noise Degradation Curves
  Fig 8: Budget x Strategy Heatmap

Usage:
    python scripts_v2/visualize.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from cbm_v2.config import RESULTS_DIR, FIGURE_DIR, TORCH_LOAD_KWARGS

# ---------------------------------------------------------------------------
# Matplotlib global settings
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
    }
)

# ---------------------------------------------------------------------------
# Color palette (matching task spec)
# ---------------------------------------------------------------------------
COLORS = {
    "Random": "#9467bd",
    "Uncertainty": "#1f77b4",
    "Importance": "#ff7f0e",
    "GreedyOracle": "#d62728",
    "ErrorTargeted": "#2ca02c",
}

MARKERS = {
    "Random": "o",
    "Uncertainty": "s",
    "Importance": "^",
    "GreedyOracle": "D",
    "ErrorTargeted": "v",
}

# Strategy key mapping (internal lowercase -> display title-case)
STRATEGY_MAP = {
    "random": "Random",
    "uncertainty": "Uncertainty",
    "importance": "Importance",
    "greedy_oracle": "GreedyOracle",
    "error_targeted": "ErrorTargeted",
}


def load_results():
    """Load intervention results from saved checkpoint.

    Returns:
        dict with keys: c_hat, c_gt, y, yhat_cbm, yhat_oracle, W,
                        attr_names, class_names, n_concepts, exp1-4
    """
    path = RESULTS_DIR / "all_intervention_results.pth"
    print(f"[Load] Reading {path}")
    data = torch.load(path, map_location="cpu", **TORCH_LOAD_KWARGS)
    print(
        f"[Load] {len(data)} top-level keys, "
        f"exp1={list(data['exp1'].keys())}, "
        f"exp2={list(data['exp2'].keys())}, "
        f"exp3={list(data['exp3'].keys())}, "
        f"exp4={list(data['exp4'].keys())}"
    )
    return data


# ---------------------------------------------------------------------------
# Fig 1: Error Attribution Pie Chart
# ---------------------------------------------------------------------------
def plot_fig1_error_pie(data):
    """Pie chart: correct / concept error / label predictor error."""
    exp1 = data["exp1"]
    n_total = len(data["y"])
    n_correct = n_total - exp1["n_wrong"]
    n_concept_err = exp1["n_concept_error"]
    n_label_err = exp1["n_label_error"]

    # Compute accuracies for title
    y = data["y"]
    yhat_cbm = data["yhat_cbm"]
    yhat_oracle = data["yhat_oracle"]
    if isinstance(yhat_cbm, torch.Tensor):
        y = y.numpy() if isinstance(y, torch.Tensor) else y
        yhat_cbm = yhat_cbm.numpy()
        yhat_oracle = yhat_oracle.numpy()
    cbm_acc = (yhat_cbm == y).mean() * 100
    oracle_acc = (yhat_oracle == y).mean() * 100

    sizes = [n_correct, n_concept_err, n_label_err]
    labels = [
        f"Correct ({n_correct})",
        f"Concept Error ({n_concept_err})",
        f"Label Error ({n_label_err})",
    ]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    explode = (0, 0.05, 0.05)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    for t in autotexts:
        t.set_fontweight("bold")

    ax.set_title(
        f"Error Attribution\n"
        f"CBM Acc = {cbm_acc:.1f}%, Oracle Acc = {oracle_acc:.1f}%",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    save_path = FIGURE_DIR / "fig1_error_attribution_pie.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 1] Saved to {save_path}")


# ---------------------------------------------------------------------------
# Fig 3: Intervention Efficiency Curves
# ---------------------------------------------------------------------------
def plot_fig3_efficiency(data):
    """Accuracy vs k for 5 strategies with CBM baseline and Oracle bound."""
    exp2 = data["exp2"]
    summary = exp2["summary"]
    k_values = [k for k in sorted(summary["random"].keys()) if k != -1]

    # Also include k=-1 (all concepts) if present
    has_all = -1 in summary["random"]
    k_plot = list(k_values)
    if has_all:
        k_plot.append(max(k_values) + 2)  # place "all" slightly after max k

    # Compute CBM baseline and Oracle upper bound
    y = np.asarray(data["y"])
    yhat_cbm = np.asarray(data["yhat_cbm"])
    yhat_oracle = np.asarray(data["yhat_oracle"])
    cbm_baseline = (yhat_cbm == y).mean() * 100
    oracle_bound = (yhat_oracle == y).mean() * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    for strat_key, display_name in STRATEGY_MAP.items():
        accs = []
        ks = []
        for k in k_values:
            accs.append(summary[strat_key][k])
            ks.append(k)
        if has_all:
            accs.append(summary[strat_key][-1])
            ks.append(k_plot[-1])

        ax.plot(
            ks,
            accs,
            color=COLORS[display_name],
            marker=MARKERS[display_name],
            linewidth=2,
            markersize=7,
            label=display_name,
        )

    # Horizontal reference lines
    ax.axhline(
        y=cbm_baseline,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"CBM Baseline ({cbm_baseline:.1f}%)",
    )
    ax.axhline(
        y=oracle_bound,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=f"Oracle Upper Bound ({oracle_bound:.1f}%)",
    )

    ax.set_xlabel("Number of Concepts Corrected (k)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(
        "Intervention Efficiency: Accuracy vs Expert Effort",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(bottom=max(0, cbm_baseline - 10), top=min(100, oracle_bound + 5))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # Annotate "all" on x-axis if present
    if has_all:
        xticks = list(k_values) + [k_plot[-1]]
        xticklabels = [str(k) for k in k_values] + ["all"]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

    fig.tight_layout()
    save_path = FIGURE_DIR / "fig3_intervention_efficiency.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 3] Saved to {save_path}")


# ---------------------------------------------------------------------------
# Fig 4: Strategy AUC Bar Chart
# ---------------------------------------------------------------------------
def plot_fig4_auc(data):
    """Bar chart of AUC for each strategy using np.trapz."""
    exp2 = data["exp2"]
    summary = exp2["summary"]
    k_values = sorted([k for k in summary["random"].keys() if k > 0])

    aucs = {}
    for strat_key, display_name in STRATEGY_MAP.items():
        ks = np.array(k_values, dtype=float)
        accs = np.array([summary[strat_key][k] for k in k_values])
        auc_val = np.trapz(accs, ks)
        aucs[display_name] = auc_val

    names = list(aucs.keys())
    values = list(aucs.values())
    bar_colors = [COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=bar_colors, edgecolor="black", linewidth=0.5)

    # Value labels on top
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("AUC (Accuracy x k)")
    ax.set_title(
        "Strategy Efficiency: Area Under the Accuracy-k Curve",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_path = FIGURE_DIR / "fig4_strategy_auc.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 4] Saved to {save_path}")


# ---------------------------------------------------------------------------
# Fig 5: k_min Distribution Histogram
# ---------------------------------------------------------------------------
def plot_fig5_kmin(data):
    """Histogram of k_min values from experiment 3."""
    exp3 = data["exp3"]
    k_mins = np.asarray(exp3["per_sample_k"])
    mean_k = exp3["mean"]
    median_k = exp3["median"]
    k1_rate = (k_mins == 1).sum() / len(k_mins) * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    # Determine bins
    max_k = int(k_mins.max())
    bins = np.arange(0.5, max_k + 1.5, 1)

    ax.hist(
        k_mins,
        bins=bins,
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )

    # Median vertical line
    ax.axvline(
        x=median_k,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median = {median_k:.1f}",
    )

    ax.set_xlabel("k_min (Minimum Concepts Needed)")
    ax.set_ylabel("Number of Samples")
    ax.set_title(
        f"Minimal Intervention Distribution\n"
        f"Mean = {mean_k:.2f}, Median = {median_k:.1f}, "
        f"k=1 Rate = {k1_rate:.1f}%",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Integer x ticks
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    save_path = FIGURE_DIR / "fig5_kmin_distribution.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 5] Saved to {save_path}")


# ---------------------------------------------------------------------------
# Fig 7: Noise Degradation Curves
# ---------------------------------------------------------------------------
def plot_fig7_noise(data):
    """Accuracy vs noise level for 3 strategies (budget=10)."""
    exp4 = data["exp4"]
    noise_summary = exp4["noise"]["summary"]

    # Collect noise levels from the data
    noise_levels = sorted(noise_summary["random"].keys())

    fig, ax = plt.subplots(figsize=(8, 5))

    for strat_key in ["random", "uncertainty", "importance"]:
        display_name = STRATEGY_MAP[strat_key]
        nl_arr = np.array(noise_levels)
        acc_arr = np.array([noise_summary[strat_key][nl] for nl in noise_levels])
        ax.plot(
            nl_arr,
            acc_arr,
            color=COLORS[display_name],
            marker=MARKERS[display_name],
            linewidth=2,
            markersize=7,
            label=display_name,
        )

    # No-intervention baseline (CBM accuracy at k=0, noise=0)
    y = np.asarray(data["y"])
    yhat_cbm = np.asarray(data["yhat_cbm"])
    cbm_baseline = (yhat_cbm == y).mean() * 100
    ax.axhline(
        y=cbm_baseline,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"No Intervention ({cbm_baseline:.1f}%)",
    )

    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(
        "Noise Degradation: Accuracy vs Expert Noise (Budget=10)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig.tight_layout()
    save_path = FIGURE_DIR / "fig7_noise_degradation.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 7] Saved to {save_path}")


# ---------------------------------------------------------------------------
# Fig 8: Budget x Strategy Heatmap
# ---------------------------------------------------------------------------
def plot_fig8_budget_heatmap(data):
    """Heatmap of accuracy for budget x strategy combinations."""
    exp4 = data["exp4"]
    budget_summary = exp4["budget"]["summary"]

    strategies = ["Random", "Uncertainty", "Importance"]
    strat_keys = ["random", "uncertainty", "importance"]
    budgets = sorted(budget_summary["random"].keys())

    # Build matrix: rows=strategies, columns=budgets
    matrix = np.zeros((len(strategies), len(budgets)))
    for i, sk in enumerate(strat_keys):
        for j, b in enumerate(budgets):
            matrix[i, j] = budget_summary[sk][b]

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, cmap="YlGn", aspect="auto")

    # Tick labels
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies)

    # Text annotations
    for i in range(len(strategies)):
        for j in range(len(budgets)):
            val = matrix[i, j]
            text_color = "white" if val < matrix.mean() else "black"
            ax.text(
                j,
                i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=text_color,
            )

    ax.set_xlabel("Budget (k)")
    ax.set_ylabel("Strategy")
    ax.set_title(
        "Budget x Strategy Accuracy Heatmap",
        fontsize=14,
        fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, label="Accuracy (%)")
    fig.tight_layout()
    save_path = FIGURE_DIR / "fig8_budget_strategy_heatmap.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 8] Saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("CBM V2 — Visualization: 6 Core Figures")
    print("=" * 60)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_results()

    plot_fig1_error_pie(data)
    plot_fig3_efficiency(data)
    plot_fig4_auc(data)
    plot_fig5_kmin(data)
    plot_fig7_noise(data)
    plot_fig8_budget_heatmap(data)

    print(f"\n[Done] All 6 figures saved to {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
