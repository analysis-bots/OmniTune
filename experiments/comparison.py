"""
Model Comparison Experiment

Compares different LLM providers across multiple settings:
- Base: One-shot baseline with base model
- Thinking: One-shot with thinking/reasoning model
- Omnitune: Full OmniTune pipeline with base model
- Random: Random baseline (random subspace + random assignments)
"""

import os
import datetime
from typing import List, Optional
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import (
    get_benchmarks,
    run_single_task,
    BENCHMARK_CATEGORIES,
    BASE_MODELS,
    THINKING_MODELS,
)


def plot_comparison(output_dir: str):
    """
    Generate comparison plots.
    Reads CSVs from output_dir and generates bar charts.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    MODEL_ORDER = ["ChatGPT", "Mistral", "Gemini"]
    SETTING_ORDER = ["Base", "Thinking", "Omnitune"]
    SETTING_COLORS = {"Base": "#1f77b4", "Thinking": "#ff7f0e", "Omnitune": "#2ca02c"}
    SETTING_HATCHES = {"Base": "", "Thinking": "///", "Omnitune": "xx"}
    MODEL_PATTERNS = {
        "ChatGPT": ("gpt-", "openai", "chatgpt"),
        "Mistral": ("mistral", "magistral"),
        "Gemini": ("gemini",),
    }
    
    def canonicalize_model(name: str) -> Optional[str]:
        lower_name = str(name).lower()
        for canonical, patterns in MODEL_PATTERNS.items():
            if any(p in lower_name for p in patterns):
                return canonical
        return None
    
    def coerce_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
        col_data = frame[column_name]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
        return pd.Series(col_data, name=column_name)
    
    LEGEND_HANDLES = [
        Patch(facecolor=SETTING_COLORS[s], hatch=SETTING_HATCHES[s], edgecolor="black", linewidth=0.6, label=s)
        for s in SETTING_ORDER
    ]
    RANDOM_HANDLE = Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Random")
    
    for mode in ["optimality", "success"]:
        if mode == "optimality":
            csv_path = Path(output_dir) / "optimality_results.csv"
            save_dir = Path(output_dir) / "charts" / "model_comparisons" / "optimality"
            ylabel = "Optimality Score"
        else:
            csv_path = Path(output_dir) / "pass_at_1_results.csv"
            save_dir = Path(output_dir) / "charts" / "model_comparisons" / "success"
            ylabel = "Success Rate"
        
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping {mode} plots")
            continue
        
        save_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(csv_path)
        
        model_col = next((c for c in df.columns if "model" in c.lower()), None)
        setting_col = next((c for c in df.columns if "setting" in c.lower()), None)
        
        if not model_col or not setting_col:
            print(f"Warning: Could not find model/setting columns in {csv_path}")
            continue
        
        benchmarks = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        
        for metric in benchmarks[:4]:
            model_series = coerce_series(df, model_col)
            setting_series = coerce_series(df, setting_col)
            metric_series = df[metric]
            canonical_series = model_series.apply(canonicalize_model)
            random_mask = model_series.fillna("").str.lower().str.contains("random")
            valid_mask = canonical_series.notna() & setting_series.isin(SETTING_ORDER)
            
            working = pd.DataFrame({
                model_col: canonical_series[valid_mask],
                setting_col: setting_series[valid_mask],
                metric: metric_series[valid_mask],
            })
            
            pivot = (
                working
                .pivot_table(index=model_col, columns=setting_col, values=metric, aggfunc="mean")
                .reindex(index=MODEL_ORDER, columns=SETTING_ORDER)
            )
            
            fig, ax = plt.subplots(figsize=(6, 4))
            n_models, n_settings = len(MODEL_ORDER), len(SETTING_ORDER)
            x = np.arange(n_models)
            width = 0.22
            
            for i, setting in enumerate(SETTING_ORDER):
                offset = (i - (n_settings - 1) / 2) * (width + 0.02)
                ax.bar(x + offset, pivot[setting], width, label=setting,
                       color=SETTING_COLORS[setting], edgecolor='black', linewidth=0.6,
                       hatch=SETTING_HATCHES[setting], zorder=3)
            
            if random_mask.any():
                random_value = float(metric_series[random_mask].mean())
                ax.axhline(random_value, color="black", linestyle="--", linewidth=1.2, zorder=4)
            
            ax.set_xticks(x)
            ax.set_xticklabels(pivot.index.tolist(), fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_facecolor('#f2f2f2')
            ax.grid(axis='y', color='white', linewidth=1)
            fig.tight_layout()
            
            metric_text = metric.lower().replace("-", "_").replace(" ", "_")
            fig.savefig(save_dir / f"{metric_text}.pdf", bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {save_dir / f'{metric_text}.pdf'}")
        
        # Save legend
        legend_fig, legend_ax = plt.subplots(figsize=(4.8, 0.45))
        legend_ax.axis('off')
        legend_ax.legend(handles=LEGEND_HANDLES + [RANDOM_HANDLE], loc='center', ncol=4, fontsize=10)
        legend_fig.savefig(save_dir / "legend_only.pdf", bbox_inches='tight')
        plt.close(legend_fig)


def run_comparison(args):
    """Run model comparison across providers and settings."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"comparison_{timestamp}"
    benchmarks = get_benchmarks(args.benchmarks)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build run configurations
    run_configs = []
    
    # Base setting (one-shot)
    for model_key, model_config in BASE_MODELS.items():
        run_configs.append({
            "canonical": model_key.capitalize() if model_key != "chatgpt" else "ChatGPT",
            "setting": "Base",
            "provider": model_config["provider"],
            "model": model_config["model"],
            "one_shot": True,
        })
    
    # Thinking setting (one-shot)
    for model_key, model_config in THINKING_MODELS.items():
        run_configs.append({
            "canonical": model_key.capitalize() if model_key != "chatgpt" else "ChatGPT",
            "setting": "Thinking",
            "provider": model_config["provider"],
            "model": model_config["model"],
            "one_shot": True,
        })
    
    # Omnitune setting (full pipeline)
    for model_key, model_config in BASE_MODELS.items():
        run_configs.append({
            "canonical": model_key.capitalize() if model_key != "chatgpt" else "ChatGPT",
            "setting": "Omnitune",
            "provider": model_config["provider"],
            "model": model_config["model"],
            "one_shot": False,
        })
    
    # Random baseline
    run_configs.append({
        "canonical": "Random",
        "setting": "Random",
        "provider": BASE_MODELS["chatgpt"]["provider"],
        "model": BASE_MODELS["chatgpt"]["model"],
        "one_shot": False,
        "random": True,
    })
    
    print(f"\n[MODEL COMPARISON]")
    print(f"  Run name: {run_name}")
    print(f"  Benchmarks: {list(benchmarks.keys())}")
    print(f"  Model configurations: {len(run_configs)}")
    print(f"  Iterations per task: {args.iterations}")
    print(f"\n  Configurations:")
    for i, cfg in enumerate(run_configs):
        print(f"    {i+1}. {cfg['canonical']} / {cfg['setting']} ({cfg['provider']}: {cfg['model']})")
    
    # Results storage
    results = defaultdict(dict)
    
    for config_idx, config in enumerate(run_configs):
        canonical = config["canonical"]
        setting = config["setting"]
        is_random = config.get("random", False)
        
        print(f"\n{'='*60}")
        print(f"[{config_idx+1}/{len(run_configs)}] {canonical} / {setting}")
        print(f"{'='*60}")
        print(f"  Provider: {config['provider']}")
        print(f"  Model: {config['model']}")
        print(f"  One-shot: {config.get('one_shot', False)}")
        print(f"  Random baseline: {is_random}")
        print()
        
        log_dir = f"logs/{run_name}_{canonical}_{setting}"
        
        for bench_name, bench_config in benchmarks.items():
            tasks = bench_config["tasks"]
            epsilon = bench_config["epsilon"]
            gt = bench_config["gt"]
            max_dist = bench_config["max_dist"]
            is_having = (bench_name == "complex")
            
            total_success, total_runs = 0, 0
            task_optimalities = []  # Store optimality per task
            
            print(f"  [{bench_name}] Running {len(tasks)} tasks x {args.iterations} iterations...")
            
            for task_idx, task in enumerate(tasks):
                task_best_dist = float('inf')
                task_successes = 0
                
                for i in range(args.iterations):
                    print(f"    {task.name} [{task_idx+1}/{len(tasks)}] iter {i+1}/{args.iterations}...", end=" ", flush=True)
                    _, dist, tokens, success = run_single_task(
                        task, epsilon,
                        model_provider=config["provider"],
                        model_name=config["model"],
                        one_shot_mode=config.get("one_shot", False),
                        assignment_lm_only=is_random,
                        subspace_lm_only_random=is_random,
                        is_having=is_having,
                        log_dir=log_dir,
                        seed=args.seed + i,
                    )
                    status = "OK" if success else "FAIL"
                    print(f"{status} (dist={dist:.4f})")
                    
                    total_runs += 1
                    if success:
                        total_success += 1
                        task_successes += 1
                    if dist < task_best_dist:
                        task_best_dist = dist
                
                # Compute optimality for this task using correct formula
                if task_best_dist == float('inf'):
                    task_optimality = 0.0
                else:
                    task_optimality = (max_dist.iloc[task_idx] - task_best_dist) / (max_dist.iloc[task_idx] - gt.iloc[task_idx] + 1e-9)
                task_optimalities.append(task_optimality)
                
                print(f"    -> {task.name}: {task_successes}/{args.iterations} success, optimality={task_optimality:.4f}")
            
            success_rate = total_success / total_runs if total_runs > 0 else 0.0
            avg_optimality = sum(task_optimalities) / len(task_optimalities) if task_optimalities else 0.0
            
            print(f"  [{bench_name}] Summary: {total_success}/{total_runs} success ({success_rate*100:.1f}%), avg_optimality={avg_optimality:.4f}\n")
            
            results[(canonical, setting)][bench_name] = {
                "success_rate": success_rate,
                "avg_optimality": avg_optimality,
            }
    
    # Build output DataFrames
    print("\n[BUILDING RESULTS]")
    rows_success, rows_optimality = [], []
    for (canonical, setting), bench_results in results.items():
        row_s = {"Model": canonical, "Setting": setting}
        row_o = {"Model": canonical, "Setting": setting}
        for bench_name in BENCHMARK_CATEGORIES.keys():
            if bench_name in bench_results:
                row_s[bench_name.replace("_", "-").title()] = bench_results[bench_name]["success_rate"]
                opt = bench_results[bench_name]["avg_optimality"]
                row_o[bench_name.replace("_", "-").title()] = opt if opt < float('inf') else None
        rows_success.append(row_s)
        rows_optimality.append(row_o)
    
    success_df = pd.DataFrame(rows_success)
    optimality_df = pd.DataFrame(rows_optimality)
    
    # Save to output_dir (compatible with compare_models_new.py)
    success_path = os.path.join(args.output_dir, "pass_at_1_results.csv")
    optimality_path = os.path.join(args.output_dir, "optimality_results.csv")
    success_df.to_csv(success_path, index=False)
    optimality_df.to_csv(optimality_path, index=False)
    
    print(f"\n[SAVED] {success_path}")
    print(f"[SAVED] {optimality_path}")
    
    print("\n[RESULTS SUMMARY - Success Rate]")
    print(success_df.to_string(index=False))
    print("\n[RESULTS SUMMARY - Optimality]")
    print(optimality_df.to_string(index=False))
    
    # Generate plots if requested
    if args.plot:
        print("\n[PLOTTING] Generating comparison plots...")
        plot_comparison(args.output_dir)
        print("[PLOTTING] Done!")


