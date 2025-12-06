import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib.ticker import FuncFormatter
from matplotlib import rcParams

rcParams.update({
    # "text.usetex": True,
    'font.family': 'Helvetica',
    # "font.serif": ["Times"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    # use mathptmx for Times + math consistency
    "text.latex.preamble": r"\usepackage{mathptmx}",
})
FONTSIZE = 34

BENCHMARKS = ["top_k", "range", "diversity", "complex"]
COLORS = plt.cm.tab10.COLORS
MARKERS = ['o', 's', 'D', '^', 'v']  # circle, square, diamond, tri_up, tri_down


def _k_formatter(y, _):
    """Format large y ticks as thousands (×1K)."""
    try:
        return f"{int(y/1000)}"
    except Exception:
        return str(y)


def plot_stress_test_split(
    benchmarks,
    value_column="Optimality",
    csv_dir="../experiments/exports",
):

    """
    For each benchmark, generate TWO separate figures and save them as PDFs:
      1) subspace chart -> {out_root}/subspace/{bench}.pdf
      2) refinement chart -> {out_root}/refinement/{bench}.pdf

    Directories are created if they don't exist.

    Parameters
    ----------
    benchmarks : Iterable[str]
        Names of benchmarks. Each will be sanitized to a filename-friendly form.
    csv_dir : str or Path
        Directory containing the CSV files named:
        average_tokens_stress_test_{bench}.csv
        (where {bench} is the sanitized benchmark name used below)
    out_root : str or Path
        Root output directory for charts.
    """

    # --- Style setup ---
    plt.rcParams.update({
        "font.family": "Helvetica",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 10,
        "xtick.labelsize": FONTSIZE-22,
        "ytick.labelsize": FONTSIZE-2,
        "axes.linewidth": 1.0,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })

    out_root = f"../../exports/charts/{value_column.lower()}"

    # csv_dir = Path(csv_dir)
    out_root = Path(out_root)
    subspace_dir = out_root / "subspace"
    refinement_dir = out_root / "refinement"

    # Ensure output directories exist
    subspace_dir.mkdir(parents=True, exist_ok=True)
    refinement_dir.mkdir(parents=True, exist_ok=True)


    for bench_name in benchmarks:
        bench = bench_name.replace("-", "_").lower()

        # Read data
        df = pd.read_csv(f"../../exports/parameter_test_results/param_test_{bench}.csv")

        # ----------------------------
        # Figure 1: Avg Num Tokens vs Refinements (per Subspace)
        # ----------------------------
        fig1, ax1 = plt.subplots(figsize=(8.5, 5))
        for i, max_subspace_iters in enumerate([1, 3, 5, 7, 10]):
            subset = df[df['Subspaces'] == max_subspace_iters]
            ax1.plot(
                subset['Refinements'],
                subset[value_column],
                label=(
                    f"{max_subspace_iters} subspaces"
                    if max_subspace_iters > 1 else f"{max_subspace_iters} subspace"
                ),
                linewidth=4,
                color=COLORS[i % len(COLORS)],
                marker=MARKERS[i % len(MARKERS)],
                markersize=12,
                markeredgecolor=COLORS[i % len(COLORS)],
            )
        ax1.set_xlabel('Refinements per subspace (K)', fontsize=FONTSIZE)
        if value_column == "Success":
            ax1.set_ylabel('Success Rate', fontsize=FONTSIZE)
            ax1.set_yticks(np.arange(0, 1.01, 0.2))
        elif value_column == "Optimality":
            ax1.set_ylabel('Optimality', fontsize=FONTSIZE)
            ax1.set_yticks(np.arange(0, 1.01, 0.2))
        elif value_column == "Tokens":
            ax1.set_ylabel('Num Tokens (x1K)', fontsize=FONTSIZE)
            ax1.yaxis.set_major_formatter(FuncFormatter(_k_formatter))

        ax1.set_xlim(0, 10)
        ax1.set_xticks([1, 3, 5, 7, 10])

        ax1.grid(True, color='black', linewidth=1.2)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(axis='both', labelsize=FONTSIZE-10)  # both x and y axes
        fig1.tight_layout(pad=1.2)

        # Save fig1
        out_path1 = refinement_dir / f"{bench}.pdf"
        fig1.savefig(out_path1, format="pdf", bbox_inches=None)
        plt.close(fig1)

        # ----------------------------
        # Figure 2: Avg Num Tokens vs Subspaces (per Refinement)
        # ----------------------------
        fig2, ax2 = plt.subplots(figsize=(8.5, 5))
        for i, max_actor_attempts in enumerate([1, 3, 5, 7, 10]):
            subset = df[df['Refinements'] == max_actor_attempts]
            ax2.plot(
                subset['Subspaces'],
                subset[value_column],
                # subset['Optimality'],
                label=(
                    f"{max_actor_attempts} refinements"
                    if max_actor_attempts > 1 else f"{max_actor_attempts} refinement"
                ),
                linewidth=4,
                color=COLORS[(i + 5) % len(COLORS)],
                marker=MARKERS[i % len(MARKERS)],
                markersize=12,
                markeredgecolor=COLORS[(i + 5) % len(COLORS)],
            )
        ax2.set_xlabel('Horizon T', fontsize=FONTSIZE)
        if value_column == "Success":
            ax2.set_ylabel('Success Rate', fontsize=FONTSIZE)
            ax2.set_yticks(np.arange(0, 1.01, 0.2))
        elif value_column == "Optimality":
            ax2.set_ylabel('Optimality', fontsize=FONTSIZE)
            ax2.set_yticks(np.arange(0, 1.01, 0.2))
        elif value_column == "Tokens":
            ax2.set_ylabel('Num Tokens (x1K)', fontsize=FONTSIZE)
            ax2.yaxis.set_major_formatter(FuncFormatter(_k_formatter))

        ax2.set_xticks([1, 3, 5, 7, 10])

        ax2.grid(True, color='black', linewidth=1.2)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(axis='both', labelsize=FONTSIZE-10)  # both x and y axes
        fig2.tight_layout(pad=1.2)

        # Save fig2
        out_path2 = subspace_dir / f"{bench}.pdf"
        fig2.savefig(out_path2, format="pdf", bbox_inches=None)
        plt.close(fig2)

        print(f"Saved: {out_path1} and {out_path2}")

    # Create dummy lines for the legend

def plot_legend_only():
    handles = []
    labels = ['T=1', 'T=3', 'T=5', 'T=7', 'T=10']
    for label, color, marker in zip(labels, COLORS, MARKERS):
        line, = plt.plot([], [], color=color, marker=marker,
                         markersize=8, linewidth=2, label=label)
        handles.append(line)

    # Create a blank figure just for the legend
    fig = plt.figure(figsize=(4.8, 0.4))
    fig.legend(
        handles, labels,
        loc='center', ncol=5,
        fontsize=16,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.4,
        borderaxespad=0.2,
    )

    # Remove all axes and whitespace
    plt.axis('off')

    # Save or show
    plt.savefig("../../exports/charts/refinement/legend_only.pdf", bbox_inches='tight')
    plt.close()
    # Create dummy lines for the legend
    handles = []
    labels = ['K=1', 'K=3', 'K=5', 'K=7', 'K=10']
    for label, color, marker in zip(labels, COLORS[5:], MARKERS):
        line, = plt.plot([], [], color=color, marker=marker,
                         markersize=12, linewidth=4, label=label)
        handles.append(line)

    # Create a blank figure just for the legend
    fig = plt.figure(figsize=(4.8, 0.4))
    fig.legend(
        handles, labels,
        loc='center', ncol=5,
        fontsize=20,
        handlelength=0.9,
        handletextpad=0.3,
        columnspacing=0.3,
        borderaxespad=0.1,
    )

    # Remove all axes and whitespace
    plt.axis('off')

    # Save or show
    plt.savefig("../../exports/charts/subspace/legend_only.pdf", bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    plot_stress_test_split(BENCHMARKS, value_column="Optimality")
    plot_stress_test_split(BENCHMARKS, value_column="Success")
    plot_stress_test_split(BENCHMARKS, value_column="Tokens")
    plot_legend_only()