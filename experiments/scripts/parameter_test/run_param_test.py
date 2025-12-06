"""
One-Shot LLM Baseline Runner

This script demonstrates how to run the one-shot LLM baseline using the existing benchmark framework.
The one-shot baseline provides the LLM with all task context and asks for a single solution
from the entire search space without any iterative refinement.

Usage:
    python run_one_shot_baseline.py

To modify which benchmarks to run, edit the bench_params list below.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from experiments.bench_marks import (
    T1a, T1b, T2a, T2b, T3a, T3b, T4a, T4b,  # Top-K benchmarks
    T5a, T5b, T6a, T6b, T7a, T7b, T8a, T8b,  # Range benchmarks
    T9a, T9b, T10a, T10b, T11a, T11b, T12a, T12b,  # Diversity benchmarks
    T13a, T13b, T14a, T14b, T15a, T15b, T16a, T16b  # Complex benchmarks
)
from opro.opro_main_loop import run_task

# Define benchmark suites
top_k_bench = [T1a, T1b, T2a, T2b, T3a, T3b, T4a, T4b]
range_bench = [T5a, T5b, T6a, T6b, T7a, T7b, T8a, T8b]
diversity_bench = [T9a, T9b, T10a, T10b, T11a, T11b, T12a, T12b]
complex_bench = [T13a, T13b, T14a, T14b, T15a, T15b, T16a, T16b]


gt_top_k = pd.Series([0.36, 0.5, 0.5, 0.03, 0.16, 0.06, 0.08, 0.14])
max_dist_top_k = pd.Series([3, 3, 3, 3, 2, 2, 3, 3])

gt_range = pd.Series([0.01, 0.02, 0.503, 0.346, 0.11, 0.04, 0.125, 0.1])
max_dist_range = pd.Series([1, 1, 1, 1, 1, 1, 1, 1])

gt_diversity = pd.Series([0.05, 0.08, 0.03333, 0.06, 0.06666667, 0.05, 0.17, 0.25])
max_dist_diversity = pd.Series([1, 1, 1, 1, 1, 1, 1, 1])

gt_complex = pd.Series([0, 0, 0, 0, 0, 0, 0, 0])
max_dist_complex = pd.Series([2, 2, 2, 2, 3, 4, 4, 4])


# Configure one-shot baseline runs
RUN_NAME = "param_test"

PATH_PREFIX = "../../exports/parameter_test_results"

NUM_ITERATIONS = 5

BENCH_PARAMS = [
    # Uncomment the benchmarks you want to run
    {"run_name": f"{RUN_NAME}_top_k", "epsilon": 0.4, "tasks": top_k_bench},
    {"run_name": f"{RUN_NAME}_range", "epsilon": 0.05, "tasks": range_bench},
    {"run_name": f"{RUN_NAME}_diversity", "epsilon": 0.0, "tasks": diversity_bench},
    {"run_name": f"{RUN_NAME}_complex", "epsilon": 0.2, "tasks": complex_bench},
]


def produce_result_csvs():
    # Run benchmarks
    for bench_param in BENCH_PARAMS:
        tasks = bench_param["tasks"]
        print("Starting new benchmark run: ", bench_param["run_name"])
        ASSIGNMENT_LM_ONLY = bench_param.get("assignment_lm_only", False)
        SUBSPACE_LM_ONLY_RANDOM = bench_param.get("subspace_lm_only_random", False)
        ONE_SHOT_MODE = bench_param.get("one_shot", False)
        OUT_PATH = f"{PATH_PREFIX}/{bench_param['run_name']}.csv"
        results = pd.DataFrame(columns=[
            "Task", "Subspaces", "Refinements", "best_dist", "num_success", "avg_token_cost"
        ])
        LOG_DIR = f"logs/log_dir_{bench_param['run_name']}"
        epsilon = bench_param["epsilon"]
        is_having = SUBSPACE_LM_ONLY_RANDOM and "complex" in bench_param["run_name"]

        for max_subspace_iters in [1, 3, 5, 7, 10]:
            for max_assignments_per_subspace in [1, 3, 5, 7, 10]:
                for i, task in enumerate(tasks):
                    best_refined_query = None
                    best_distance = float('inf')
                    total_token_use = 0
                    num_successful_refinements = 0
                    # epsilon = epsilons[i // 8]
                    for j in range(NUM_ITERATIONS):
                        print(f"Running task {task.name}, iteration {j+1}/{NUM_ITERATIONS}, epsilon {epsilon}")
                        refined_query, refinement_distance, overall_token_use = run_task(task, epsilon,
                                                                                         perform_analysis=False,
                                                                                         assignment_lm_only_mode=ASSIGNMENT_LM_ONLY,
                                                                                         subspace_lm_only_random=SUBSPACE_LM_ONLY_RANDOM,
                                                                                         one_shot_mode=ONE_SHOT_MODE,
                                                                                         is_having=is_having,
                                                                                         log_dir=LOG_DIR,
                                                                                         max_assignments_per_subspace=max_assignments_per_subspace,
                                                                                         max_subspace_iters=max_subspace_iters)
                        total_token_use += overall_token_use
                        if refinement_distance < float('inf'):
                            num_successful_refinements += 1
                        if refinement_distance < best_distance:
                            best_distance = refinement_distance
                            best_refined_query = refined_query
                    avg_token_use = total_token_use / NUM_ITERATIONS
                    # pandas_dict['refined_query'].append(best_refined_query)  # top @ 5 refinements with lowest distance
                    results.loc[len(results)] = {
                        "Task": task.name,
                        "Subspaces": max_subspace_iters,
                        "Refinements": max_assignments_per_subspace,
                        "best_dist": best_distance,
                        "num_success": num_successful_refinements,
                        "avg_token_cost": avg_token_use
                    }

        results = results.groupby(['T', 'K'], as_index=False).agg({
            'best_dist': 'mean',
            'num_success': 'mean',
            'avg_token_cost': 'mean'
        })
        results.to_csv(OUT_PATH, index=False)
        print(results.head().to_string(index=False))


def post_process_results():
    bench_dict = {"Top-k": {"df": f"{PATH_PREFIX}/{RUN_NAME}_top_k.csv", "gt": gt_top_k, "max_dist": max_dist_top_k},
                  "Range": {"df": f"{PATH_PREFIX}/{RUN_NAME}_range.csv", "gt": gt_range, "max_dist": max_dist_range},
                  "Diversity": {"df": f"{PATH_PREFIX}/{RUN_NAME}_diversity.csv", "gt": gt_diversity, "max_dist": max_dist_diversity},
                  "Complex": {"df": f"{PATH_PREFIX}/{RUN_NAME}_complex.csv", "gt": gt_complex, "max_dist": max_dist_complex}}
    for bench_name, bench_data in bench_dict.items():
        df = pd.read_csv(bench_data["df"])
        ground_truth = bench_data["gt"]
        max_dist = bench_data["max_dist"]
        df["Success"] = df["num_success"] / NUM_ITERATIONS
        df["Optimality"] = (max_dist - df["best_dist"]) / (max_dist - ground_truth)
        df["Tokens"] = df["avg_token_cost"]
        df.to_csv(bench_data["df"], index=False)


if __name__ == '__main__':
    produce_result_csvs()
    post_process_results()
