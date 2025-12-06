# Adjusted version: 3 x-ticks (ChatGPT, Mistral, Gemini) and wider bars.

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# MODE = "success_rate"
MODE = "optimality"


# Config
if MODE == "optimality":
    csv_path = Path("../optimality_results.csv")
    save_dir = Path("../exports/model_comparisons/optimality")
else:
    csv_path = Path("../pass_at_1_results.csv")
    save_dir = Path("../exports/model_comparisons/success")
save_dir.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["ChatGPT", "Mistral", "Gemini"]
SETTING_ORDER = ["Base", "Thinking", "Omnitune"]
SETTING_COLORS = {"Base": "#1f77b4", "Thinking": "#ff7f0e", "Omnitune": "#2ca02c"}
SETTING_HATCHES = {"Base": "", "Thinking": "///", "Omnitune": "xx"}
MODEL_PATTERNS = {
    "ChatGPT": ("gpt-", "openai", "chatgpt"),
    "Mistral": ("mistral", "magistral"),
    "Gemini": ("gemini",),
}
RANDOM_HANDLE = Line2D(
    [0],
    [0],
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Random",
)

LEGEND_HANDLES = [
    Patch(
        facecolor=SETTING_COLORS[setting],
        hatch=SETTING_HATCHES[setting],
        edgecolor="black",
        linewidth=0.6,
        label=setting,
    )
    for setting in SETTING_ORDER
]

def coerce_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
    """Return the first 1-D column matching `column_name`, even if duplicates exist."""
    col_data = frame[column_name]
    if isinstance(col_data, pd.DataFrame):
        col_data = col_data.iloc[:, 0]
    return pd.Series(col_data, name=column_name)


def canonicalize_model(name: str) -> str | None:
    """Map detailed model names to canonical groups used for plotting."""
    lower_name = str(name).lower()
    for canonical, patterns in MODEL_PATTERNS.items():
        if any(p in lower_name for p in patterns):
            return canonical
    return None


df = pd.read_csv(csv_path)
model_col = next(c for c in df.columns if "model" in c.lower() or "provider" in c.lower())
setting_col = next(c for c in df.columns if any(k in c.lower() for k in ["setting", "variant", "mode", "method"]))
if setting_col == model_col:
    setting_candidates = [
        c
        for c in df.columns
        if c != model_col and any(k in c.lower() for k in ["setting", "variant", "mode", "method"])
    ]
    if setting_candidates:
        setting_col = setting_candidates[0]
    else:
        raise ValueError("Could not infer a setting column distinct from the model column.")
benchmarks = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

for metric in benchmarks[:4]:
    model_series = coerce_series(df, model_col)
    setting_series = coerce_series(df, setting_col)
    metric_series = df[metric]
    canonical_series = model_series.apply(canonicalize_model)
    random_mask = model_series.fillna("").str.lower().str.contains("random")
    valid_mask = canonical_series.notna() & setting_series.isin(SETTING_ORDER)
    working = pd.DataFrame(
        {
            model_col: canonical_series[valid_mask],
            setting_col: setting_series[valid_mask],
            metric: metric_series[valid_mask],
        }
    )
    pivot = (
        working
        .pivot_table(index=model_col, columns=setting_col, values=metric, aggfunc="mean")
        .reindex(index=MODEL_ORDER, columns=SETTING_ORDER)
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    n_models, n_settings = len(MODEL_ORDER), len(SETTING_ORDER)
    x = np.arange(n_models)
    width = 0.22  # wider bars

    for i, setting in enumerate(SETTING_ORDER):
        offset = (i - (n_settings - 1) / 2) * (width + 0.02)
        color = SETTING_COLORS[setting]
        ax.bar(
            x + offset,
            pivot[setting],
            width,
            label=setting,
            color=color,
            edgecolor='black',
            linewidth=0.6,
            hatch=SETTING_HATCHES[setting],
            zorder=3,
        )

    if random_mask.any():
        random_value = float(metric_series[random_mask].mean())
        ax.axhline(
            random_value,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Random",
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), fontsize=26)
    ax.set_yticklabels(np.round(ax.get_yticks(), 1), fontsize=22)
    if MODE == "success_rate":
        ax.set_ylabel("Success Rate", fontsize=26)
    else:
        ax.set_ylabel("Optimality Score", fontsize=26)
    ax.set_facecolor('#f2f2f2')
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='white', linewidth=1)

    # ax.legend(
    #     handles=LEGEND_HANDLES + [RANDOM_HANDLE],
    #     loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=4,
    #     fontsize=11, handlelength=1.8, handletextpad=0.6, columnspacing=0.8,
    #     frameon=True, facecolor='white', framealpha=0.9
    # )

    # ax.set_title(metric, fontsize=14, pad=10)
    fig.tight_layout()
    metric_text = metric.lower().replace("-", "_").replace("div ", "").replace(" queries", "")
    fig.savefig(save_dir / f"{metric_text}.pdf", bbox_inches='tight')
    plt.close(fig)

legend_fig, legend_ax = plt.subplots(figsize=(4.8, 0.45))
legend_ax.axis('off')
legend_ax.legend(
    handles=LEGEND_HANDLES + [RANDOM_HANDLE],
    loc='center',
    ncol=4,
    fontsize=26,
    handlelength=1.6,
    handletextpad=0.5,
    columnspacing=0.7,
    frameon=True,
    facecolor='white',
    framealpha=0.9,
)
legend_fig.savefig(save_dir / "legend_only.pdf", bbox_inches='tight')
plt.close(legend_fig)

print("Saved grouped bar charts with 3 model ticks and wider bars.")
