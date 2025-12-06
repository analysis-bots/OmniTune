# Load CSV, infer columns, and plot 3-model color groups in Matplotlib (pyplot) with error bars if present.
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SUCCESS = True

plt.rcParams.update({
    'font.family': 'Helvetica',
    'font.size': 12,
    'legend.fontsize': 11,
    "text.latex.preamble": r"\usepackage{mathptmx}",
})# Load the user's data
csv_path = Path("../pass_at_1_results.csv") if SUCCESS else Path("../optimality_results.csv")
df_dirty = pd.read_csv(csv_path)

# Detect model column
model_col = [c for c in df_dirty.columns if "model" in c.lower()]
model_col = model_col[0] if model_col else df_dirty.columns[0]

# Separate the "random" row if it exists (case-insensitive match)
random_row = df_dirty[df_dirty[model_col].str.lower() == "random"]
df = df_dirty[df_dirty[model_col].str.lower() != "random"].reset_index(drop=True)

# Detect benchmark columns — assume each benchmark has separate score columns (e.g., "Div Top K", "Range", etc.)
# Keep only numeric columns that could be benchmarks
benchmarks = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if len(benchmarks) == 0:
    raise ValueError("No numeric benchmark columns detected in CSV.")

# Identify model column
model_col = None
for c in df.columns:
    if "model" in c.lower():
        model_col = c
        break
if model_col is None:
    model_col = df.columns[0]

# === Define your color mapping per trio ===
colors_map = ["#a0db8e", "#ffa500", "#0a75ad"]  #  OpenAI green, Mistral orange, Google blue,
labels_map = ["ChatGPT", "Mistral", "Gemini"]

# Assign colors cyclically per 3 models
colors = [colors_map[i // 3 % len(colors_map)] for i in range(len(df))]

# Plot 4 subplots (or fewer if fewer benchmarks exist)
n_bench = min(4, len(benchmarks))
fig, axes = plt.subplots(1, n_bench, figsize=(9 * n_bench, 2.7), sharey=False)

if n_bench == 1:
    axes = [axes]

labels = ['Base', 'Thinking', 'Ours'] * 3
save_dir = Path("../exports/model_comparisons/success") if SUCCESS else Path("../exports/model_comparisons/optimality")


save_dir.mkdir(parents=True, exist_ok=True)


for i, bench in enumerate(benchmarks[:n_bench]):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis='y', color='white', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    ax.grid(axis='x', visible=False)
    bars = ax.bar(df[model_col], df[bench], color=colors)
    bold_labels = []  # to store bolded tick labels

    # Highlight every 3rd bar (2, 5, 8, …)
    for j, bar in enumerate(bars):
        if (j + 1) % 3 == 0:  # every 3rd bar
            bar.set_edgecolor("black")
            bar.set_linewidth(1.5)
            bold_labels.append(True)
        else:
            bar.set_edgecolor("none")
            bold_labels.append(False)


    # ax.set_title(f"{bench} Benchmark Success Rate", fontsize=20)
    # ax.set_title(f"{bench} Benchmark Avg Optimality", fontsize=16)
    # ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
    ax.set_xlabel("Setting", fontsize=30)

    # Format labels (bold for our models)
    xticks = range(len(df[model_col]))
    ax.set_xticks(xticks)
    new_labels = []
    for j, name in enumerate(labels):
        if bold_labels[j]:
            new_labels.append(f"$\\bf{{{name}}}$")  # LaTeX bold
        else:
            new_labels.append(name)
    ax.set_xticklabels(new_labels, rotation=40, ha='right', fontsize=20)
    if SUCCESS:
        ax.set_ylabel("Success Rate", fontsize=30)
    else:
        ax.set_ylabel("Optimality Score", fontsize=30)

    # Add dashed horizontal line for the "random" baseline (if exists)
    if not random_row.empty:
        y = random_row[bench].iloc[0]
        ax.axhline(y=y, color="black", linestyle="--", linewidth=1, label="Random")

    ax.tick_params(axis='y', labelsize=24)
    # gray background + only horizontal white grid lines
    ax.set_facecolor("#f0f0f0")

    # add legend per subplot
    # Add dashed line for "Random"
    random_handle = Line2D(
        [0], [0],
        color="black",
        linestyle="--",
        linewidth=1,
        label="Random"
    )

    legend_handles = [Patch(facecolor=c, edgecolor='black', label=l) for c, l in zip(colors_map, labels_map)]
    # Combine them — "Random" will appear last (bottom of legend)
    legend_handles.append(random_handle)

    # ax.legend(handles=legend_handles, loc='upper left', fontsize=14, frameon=True, facecolor='white', framealpha=0.75)

    # fig.suptitle("Model Comparison Across Benchmarks", fontsize=13)
    fig.tight_layout(rect=[1, 1, 1, 1])

    # out_path_multi = Path("../plot_success_at_1.png")
    out_path_multi = save_dir / f"{bench.replace(' ', '_').replace('-', '_').lower()}.pdf"

    plt.savefig(out_path_multi, bbox_inches='tight', format='pdf')
    plt.close()

# plot the legend only

# Create dummy lines for the legend
# for label, color in zip(labels_map, colors_map):
#     line, = plt.plot([], [], color=color, linewidth=8, label=label)
#     handles.append(line)
handles = [Patch(facecolor=c, edgecolor='black', label=l) for c, l in zip(colors_map, labels_map)]

# Create a blank figure just for the legend
fig = plt.figure(figsize=(4.8, 0.4))
fig.legend(
    handles, labels_map,
    loc='center', ncol=3,
    fontsize=16,
    handlelength=1.2,
    handletextpad=0.4,
    columnspacing=0.4,
    borderaxespad=0.2,
)

# Remove all axes and whitespace
plt.axis('off')

# Save or show
plt.savefig("../exports/model_comparisons/success/legend_only.pdf", bbox_inches='tight') if SUCCESS else\
    plt.savefig("../exports/model_comparisons/optimality/legend_only.pdf", bbox_inches='tight')
plt.close()

#
# fig, ax = plt.subplots(figsize=(6, 2))
# legend_handles = [Patch(facecolor=c, edgecolor='black', label=l) for c, l in zip(colors_map, labels_map)]
# ax.legend(handles=legend_handles, loc='upper left', fontsize=14, frameon=True, facecolor='white', framealpha=0.75)
# out_path_legend = save_dir / "model_legend.pdf"
# plt.savefig(out_path_legend, bbox_inches='tight', format='pdf')

# # fig.suptitle("Model Comparison Across Benchmarks", fontsize=13)
# fig.tight_layout(rect=[1, 1, 1, 1])
#
# # out_path_multi = Path("../plot_success_at_1.png")
# out_path_multi = Path("../plot_optimality.png")
#
# plt.savefig(out_path_multi, dpi=200, bbox_inches='tight')
# plt.close()
# plt.show()
