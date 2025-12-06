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
import time

from mistralai import models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from bench_marks import (
    T1a, T1b, T2a, T2b, T3a, T3b, T4a, T4b,  # Top-K benchmarks
    T5a, T5b, T6a, T6b, T7a, T7b, T8a, T8b,  # Range query benchmarks
    T9a, T9b, T10a, T10b, T11a, T11b, T12a, T12b,  # Diversity benchmarks
    T13a, T13b, T14a, T14b, T15a, T15b, T16a, T16b,  # Complex benchmarks
)
from opro.opro_main_loop import run_task

if __name__ == '__main__':
    # Define benchmark suites
    top_k_bench = [T1a, T1b, T2a, T2b, T3a, T3b, T4a, T4b]
    range_bench = [T5a, T5b, T6a, T6b, T7a, T7b, T8a, T8b]
    diversity_bench = [T9a, T9b, T10a, T10b, T11a, T11b, T12a, T12b]
    complex_bench = [T13a, T13b, T14a, T14b, T15a, T15b, T16a, T16b]
    
    # Configure one-shot baseline runs
    RUN_NAME = "one_shot_baseline_mistral_thinking"
    
    bench_params = [
        # Uncomment the benchmarks you want to run
        # {"run_name": f"{RUN_NAME}_top_k", "one_shot": True, "epsilon": 0.4, "tasks": top_k_bench},
        # {"run_name": f"{RUN_NAME}_range", "one_shot": True, "epsilon": 0.05, "tasks": range_bench},
        # {"run_name": f"{RUN_NAME}_diversity", "one_shot": True, "epsilon": 0.0, "tasks": diversity_bench},
        {"run_name": f"{RUN_NAME}_complex", "one_shot": True, "epsilon": 0.2, "tasks": complex_bench},
    ]
    
    # Run benchmarks
    for bench_param in bench_params:
        tasks = bench_param["tasks"]
        print("=" * 80)
        print(f"Starting ONE-SHOT baseline run: {bench_param['run_name']}")
        print("=" * 80)
        
        ONE_SHOT_MODE = bench_param.get("one_shot", False)
        OUT_PATH = f"exports/mistral/{bench_param['run_name']}.csv"
        LOG_DIR = f"logs/log_dir_{bench_param['run_name']}"
        epsilon = bench_param["epsilon"]
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        
        # Prepare results dictionary
        results = {
            'task_name': [t.name for t in tasks],
            'original_query': [t.original_query.strip() for t in tasks],
            'refined_query': [],
            'refinement_distance': [],
            'num_successful_refinements': [],
            'token_use': []
        }
        
        # Run each task (one-shot doesn't need multiple iterations, but we'll keep the structure for consistency)
        num_iterations = 5
        
        for i, task in enumerate(tasks):
            print(f"\n{'='*60}")
            print(f"Task {i+1}/{len(tasks)}: {task.name}")
            print(f"{'='*60}")
            
            best_refined_query = None
            best_distance = float('inf')
            total_token_use = 0
            num_successful_refinements = 0
            
            for j in range(num_iterations):
                print(f"Running iteration {j+1}/{num_iterations}, epsilon={epsilon}")
                try:
                    refined_query, refinement_distance, overall_token_use = run_task(
                        task,
                        epsilon,
                        perform_analysis=False,
                        one_shot_mode=ONE_SHOT_MODE,
                        log_dir=LOG_DIR,
                        max_assignments_per_subspace=1  # Not used in one-shot, but set for consistency
                    )
                except models.sdkerror.SDKError as e:
                    # timeout
                    print(f"Timeout or API error during task {task.name}, iteration {j+1}: {e}")
                    time.sleep(10)
                    refined_query, refinement_distance, overall_token_use = run_task(
                        task,
                        epsilon,
                        perform_analysis=False,
                        one_shot_mode=ONE_SHOT_MODE,
                        log_dir=LOG_DIR,
                        max_assignments_per_subspace=1  # Not used in one-shot, but set for consistency
                    )

                total_token_use += overall_token_use
                
                if refinement_distance < float('inf'):
                    num_successful_refinements += 1
                
                if refinement_distance < best_distance:
                    best_distance = refinement_distance
                    best_refined_query = refined_query
            
            # Record results
            results['refined_query'].append(best_refined_query)
            results['refinement_distance'].append(best_distance)
            results['num_successful_refinements'].append(num_successful_refinements)
            results['token_use'].append(total_token_use)
            
            print(f"\nTask {task.name} completed:")
            print(f"  Best distance: {best_distance:.4f}")
            print(f"  Successful: {num_successful_refinements}/{num_iterations}")
            print(f"  Tokens used: {total_token_use}")

        # Save results to CSV
        df = pd.DataFrame(results)
        df.to_csv(OUT_PATH, index=False)
        
        print("\n" + "=" * 80)
        print(f"Results saved to: {OUT_PATH}")
        print("=" * 80)
        print("\nSummary:")
        print(df[['task_name', 'refinement_distance', 'token_use']].to_string(index=False))
        print()
