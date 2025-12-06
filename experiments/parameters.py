"""
Parameter Test Experiment

Grid search over hyperparameters:
- T (max_subspace_iters): Number of subspace iterations
- K (max_assignments_per_subspace): Refinements per subspace
"""

import os
import datetime
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import get_benchmarks, run_single_task


def plot_parameters(output_dir: str, benchmarks: List[str]):
    """
    Generate parameter test plots.
    Reads CSVs from output_dir/parameter_test_results/ and generates line charts.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    
    COLORS = plt.cm.tab10.colors
    MARKERS = ['o', 's', 'D', '^', 'v']
    FONTSIZE = 24
    
    def _k_formatter(y, _):
        try:
            return f"{int(y/1000)}"
        except:
            return str(y)
    
    csv_dir = Path(output_dir) / "parameter_test_results"
    
    for value_column in ["Optimality", "Success", "Tokens"]:
        out_root = Path(output_dir) / "charts" / value_column.lower()
        subspace_dir = out_root / "subspace"
        refinement_dir = out_root / "refinement"
        subspace_dir.mkdir(parents=True, exist_ok=True)
        refinement_dir.mkdir(parents=True, exist_ok=True)
        
        for bench_name in benchmarks:
            bench = bench_name.replace("-", "_").lower()
            csv_path = csv_dir / f"param_test_{bench}.csv"
            
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found, skipping")
                continue
            
            df = pd.read_csv(csv_path)
            
            # Figure 1: vs Refinements (per Subspace)
            fig1, ax1 = plt.subplots(figsize=(8.5, 5))
            for i, T in enumerate([1, 3, 5, 7, 10]):
                subset = df[df['Subspaces'] == T]
                if not subset.empty:
                    ax1.plot(subset['Refinements'], subset[value_column],
                             label=f"{T} subspace{'s' if T > 1 else ''}",
                             linewidth=3, color=COLORS[i % len(COLORS)],
                             marker=MARKERS[i % len(MARKERS)], markersize=10)
            
            ax1.set_xlabel('Refinements per subspace (K)', fontsize=FONTSIZE)
            if value_column == "Tokens":
                ax1.set_ylabel('Num Tokens (x1K)', fontsize=FONTSIZE)
                ax1.yaxis.set_major_formatter(FuncFormatter(_k_formatter))
            else:
                ax1.set_ylabel(value_column, fontsize=FONTSIZE)
                ax1.set_yticks(np.arange(0, 1.01, 0.2))
            ax1.set_xticks([1, 3, 5, 7, 10])
            ax1.grid(True, color='gray', linewidth=0.5, alpha=0.5)
            fig1.tight_layout()
            fig1.savefig(refinement_dir / f"{bench}.pdf", format="pdf")
            plt.close(fig1)
            print(f"Saved: {refinement_dir / f'{bench}.pdf'}")
            
            # Figure 2: vs Subspaces (per Refinement)
            fig2, ax2 = plt.subplots(figsize=(8.5, 5))
            for i, K in enumerate([1, 3, 5, 7, 10]):
                subset = df[df['Refinements'] == K]
                if not subset.empty:
                    ax2.plot(subset['Subspaces'], subset[value_column],
                             label=f"{K} refinement{'s' if K > 1 else ''}",
                             linewidth=3, color=COLORS[(i + 5) % len(COLORS)],
                             marker=MARKERS[i % len(MARKERS)], markersize=10)
            
            ax2.set_xlabel('Horizon T', fontsize=FONTSIZE)
            if value_column == "Tokens":
                ax2.set_ylabel('Num Tokens (x1K)', fontsize=FONTSIZE)
                ax2.yaxis.set_major_formatter(FuncFormatter(_k_formatter))
            else:
                ax2.set_ylabel(value_column, fontsize=FONTSIZE)
                ax2.set_yticks(np.arange(0, 1.01, 0.2))
            ax2.set_xticks([1, 3, 5, 7, 10])
            ax2.grid(True, color='gray', linewidth=0.5, alpha=0.5)
            fig2.tight_layout()
            fig2.savefig(subspace_dir / f"{bench}.pdf", format="pdf")
            plt.close(fig2)
            print(f"Saved: {subspace_dir / f'{bench}.pdf'}")
        
        # Save legends
        handles_T = [plt.Line2D([0], [0], color=COLORS[i], marker=MARKERS[i], markersize=8, linewidth=2, label=f'T={v}')
                     for i, v in enumerate([1, 3, 5, 7, 10])]
        fig = plt.figure(figsize=(4.8, 0.4))
        fig.legend(handles_T, [f'T={v}' for v in [1, 3, 5, 7, 10]], loc='center', ncol=5, fontsize=10)
        plt.axis('off')
        fig.savefig(refinement_dir / "legend_only.pdf", bbox_inches='tight')
        plt.close(fig)
        
        handles_K = [plt.Line2D([0], [0], color=COLORS[(i+5)%len(COLORS)], marker=MARKERS[i], markersize=8, linewidth=2, label=f'K={v}')
                     for i, v in enumerate([1, 3, 5, 7, 10])]
        fig = plt.figure(figsize=(4.8, 0.4))
        fig.legend(handles_K, [f'K={v}' for v in [1, 3, 5, 7, 10]], loc='center', ncol=5, fontsize=10)
        plt.axis('off')
        fig.savefig(subspace_dir / "legend_only.pdf", bbox_inches='tight')
        plt.close(fig)


def run_parameters(args):
    """Run parameter sweep over T (subspaces) and K (refinements)."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"param_test_{timestamp}"
    benchmarks = get_benchmarks(args.benchmarks)
    
    out_dir = os.path.join(args.output_dir, "parameter_test_results")
    os.makedirs(out_dir, exist_ok=True)
    
    T_values = [int(x) for x in args.subspaces.split(",")]
    K_values = [int(x) for x in args.refinements.split(",")]
    
    total_param_combos = len(T_values) * len(K_values)
    
    print(f"\n[PARAMETER TEST]")
    print(f"  Run name: {run_name}")
    print(f"  Benchmarks: {list(benchmarks.keys())}")
    print(f"  T values (subspaces): {T_values}")
    print(f"  K values (refinements): {K_values}")
    print(f"  Total (T, K) combinations: {total_param_combos}")
    print(f"  Iterations per task: {args.iterations}")
    print(f"  Output directory: {out_dir}")
    
    benchmark_names_run = []
    
    for bench_idx, (bench_name, bench_config) in enumerate(benchmarks.items()):
        benchmark_names_run.append(bench_name)
        tasks = bench_config["tasks"]
        epsilon = bench_config["epsilon"]
        gt = bench_config["gt"]
        max_dist = bench_config["max_dist"]
        is_having = (bench_name == "complex")
        
        out_path = os.path.join(out_dir, f"param_test_{bench_name}.csv")
        log_dir = f"logs/{run_name}_{bench_name}"
        
        total_runs_for_bench = total_param_combos * len(tasks) * args.iterations
        
        print(f"\n{'='*60}")
        print(f"[{bench_idx+1}/{len(benchmarks)}] Parameter Test: {bench_name}")
        print(f"{'='*60}")
        print(f"  Tasks: {len(tasks)}")
        print(f"  Epsilon: {epsilon}")
        print(f"  Total runs for this benchmark: {total_runs_for_bench}")
        print(f"  Output: {out_path}")
        print()
        
        results = []
        combo_idx = 0
        
        for T in T_values:
            for K in K_values:
                combo_idx += 1
                print(f"  [{combo_idx}/{total_param_combos}] T={T}, K={K}")
                
                for task_idx, task in enumerate(tasks):
                    best_dist = float('inf')
                    total_tokens, num_success = 0, 0
                    
                    for i in range(args.iterations):
                        print(f"    {task.name} [{task_idx+1}/{len(tasks)}] iter {i+1}/{args.iterations}...", end=" ", flush=True)
                        _, dist, tokens, success = run_single_task(
                            task, epsilon,
                            is_having=is_having,
                            log_dir=log_dir,
                            max_subspace_iters=T,
                            max_assignments_per_subspace=K,
                            seed=args.seed + i,
                        )
                        status = "OK" if success else "FAIL"
                        print(f"{status} (dist={dist:.4f}, tokens={tokens})")
                        
                        total_tokens += tokens
                        if success:
                            num_success += 1
                        if dist < best_dist:
                            best_dist = dist
                    
                    success_rate = num_success / args.iterations
                    print(f"    -> {task.name}: {num_success}/{args.iterations} ({success_rate*100:.1f}%), best_dist={best_dist:.4f}")
                    
                    results.append({
                        "Task": task.name,
                        "Subspaces": T,
                        "Refinements": K,
                        "best_dist": best_dist,
                        "num_success": num_success,
                        "avg_token_cost": total_tokens / args.iterations,
                        "task_idx": task_idx,
                    })
                print()
        
        # Create DataFrame and aggregate
        print(f"  [PROCESSING] Computing metrics...")
        df = pd.DataFrame(results)
        
        # Compute Success and Optimality per task
        df["Success"] = df["num_success"] / args.iterations
        
        # Optimality calculation needs per-task gt and max_dist
        def compute_optimality(row):
            idx = row["task_idx"]
            if row["best_dist"] == float('inf'):
                return 0.0
            return (max_dist.iloc[idx] - row["best_dist"]) / (max_dist.iloc[idx] - gt.iloc[idx] + 1e-9)
        
        df["Optimality"] = df.apply(compute_optimality, axis=1)
        df["Tokens"] = df["avg_token_cost"]
        
        # Aggregate by (Subspaces, Refinements)
        agg_df = df.groupby(["Subspaces", "Refinements"], as_index=False).agg({
            "best_dist": "mean",
            "num_success": "mean",
            "avg_token_cost": "mean",
            "Success": "mean",
            "Optimality": "mean",
            "Tokens": "mean",
        })
        
        agg_df.to_csv(out_path, index=False)
        print(f"  [SAVED] {out_path}")
        
        print(f"\n  [RESULTS - {bench_name}]")
        print(agg_df.to_string(index=False))
    
    # Generate plots if requested
    if args.plot:
        print("\n[PLOTTING] Generating parameter test plots...")
        plot_parameters(args.output_dir, benchmark_names_run)
        print("[PLOTTING] Done!")


