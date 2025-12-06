"""
Model Comparison Runner

This script compares Omnitune across different LLM backends and settings:
- Base (one-shot): Direct LLM query without iteration
- Thinking (one-shot): Reasoning/thinking mode LLMs without iteration  
- Omnitune: Full Omnitune pipeline with the base models
- Random: Random baseline (subspace_lm_only_random=True, assignment_lm_only=True)

Outputs two CSV files:
- pass_at_1_results.csv: Success rate (constraints satisfied)
- optimality_results.csv: Average refinement distance when successful

Usage:
    python run_models_comparison.py
"""

import sys
import os
import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from bench_marks import (
    T1a, T1b, T2a, T2b, T3a, T3b, T4a, T4b,  # Top-K benchmarks
    T5a, T5b, T6a, T6b, T7a, T7b, T8a, T8b,  # Range query benchmarks
    T9a, T9b, T10a, T10b, T11a, T11b, T12a, T12b,  # Diversity benchmarks
    T13a, T13b, T14a, T14b, T15a, T15b, T16a, T16b,  # Complex benchmarks
)
from opro.opro_main_loop import run_task

# ============================================================================
# Model Configurations
# ============================================================================

# Base models (non-thinking)
BASE_MODELS = {
    "ChatGPT": {"provider": "openai", "model": "gpt-4.1-mini"},
    "Gemini": {"provider": "google", "model": "gemini-2.0-flash-lite"},
    "Mistral": {"provider": "cloudflare", "model": "@cf/mistralai/mistral-small-3.1-24b-instruct"},
}

# Thinking/reasoning models
THINKING_MODELS = {
    "ChatGPT": {"provider": "openai", "model": "gpt-5-mini"},
    "Gemini": {"provider": "google", "model": "gemini-2.5-flash-lite"},
    "Mistral": {"provider": "mistral", "model": "magistral-small-latest"},
}

# Benchmark categories
BENCHMARK_CATEGORIES = {
    "Top-K": [T1a, T1b, T2a, T2b, T3a, T3b, T4a, T4b],
    "Range": [T5a, T5b, T6a, T6b, T7a, T7b, T8a, T8b],
    "Diversity": [T9a, T9b, T10a, T10b, T11a, T11b, T12a, T12b],
    "Complex": [T13a, T13b, T14a, T14b, T15a, T15b, T16a, T16b],
}

# Epsilon values per category
EPSILON_MAP = {
    "Top-K": 0.4,
    "Range": 0.05,
    "Diversity": 0.0,
    "Complex": 0.2,
}

# ============================================================================
# Experiment Runner
# ============================================================================

def run_single_task(
    task,
    epsilon: float,
    model_provider: str,
    model_name: str,
    one_shot_mode: bool = False,
    assignment_lm_only: bool = False,
    subspace_lm_only_random: bool = False,
    log_dir: str = "logs",
    is_having: bool = False,
) -> Tuple[Optional[str], float, int, bool]:
    """
    Run a single task and return results.
    
    Returns:
        Tuple of (refined_query, refinement_distance, token_use, is_successful)
    """
    try:
        refined_query, refinement_distance, token_use = run_task(
            task,
            epsilon,
            perform_analysis=False,
            one_shot_mode=one_shot_mode,
            assignment_lm_only_mode=assignment_lm_only,
            subspace_lm_only_random=subspace_lm_only_random,
            is_having=is_having,
            log_dir=log_dir,
            random_seed=42,
            max_assignments_per_subspace=5,
            max_subspace_iters=5,
            model_provider=model_provider,
            model_name=model_name,
        )
        # Success = finite distance (constraints satisfied)
        is_successful = refinement_distance < float('inf')
        return refined_query, refinement_distance, token_use, is_successful
    except Exception as e:
        print(f"Error running task {task.name}: {e}")
        return None, float('inf'), 0, False


def run_benchmark_category(
    category_name: str,
    tasks: List,
    epsilon: float,
    model_provider: str,
    model_name: str,
    setting: str,
    num_iterations: int = 5,
    log_dir: str = "logs",
) -> Dict:
    """
    Run all tasks in a benchmark category and aggregate results.
    
    Returns:
        Dict with success_rate and avg_optimality for the category
    """
    # Determine run mode based on setting
    one_shot_mode = setting in ["Base", "Thinking"]
    assignment_lm_only = (setting == "Random")
    subspace_lm_only_random = (setting == "Random")
    is_having = (category_name == "Complex")
    
    total_successes = 0
    total_runs = 0
    successful_distances = []
    
    for task in tasks:
        for iteration in range(num_iterations):
            print(f"  [{setting}] {category_name}/{task.name} iter {iteration+1}/{num_iterations}")
            
            _, distance, _, is_successful = run_single_task(
                task=task,
                epsilon=epsilon,
                model_provider=model_provider,
                model_name=model_name,
                one_shot_mode=one_shot_mode,
                assignment_lm_only=assignment_lm_only,
                subspace_lm_only_random=subspace_lm_only_random,
                log_dir=log_dir,
                is_having=is_having,
            )
            
            total_runs += 1
            if is_successful:
                total_successes += 1
                successful_distances.append(distance)
    
    success_rate = total_successes / total_runs if total_runs > 0 else 0.0
    avg_optimality = (
        sum(successful_distances) / len(successful_distances)
        if successful_distances else float('inf')
    )
    
    return {
        "success_rate": success_rate,
        "avg_optimality": avg_optimality,
        "total_runs": total_runs,
        "total_successes": total_successes,
    }


def run_all_experiments(num_iterations: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all experiments across models, settings, and benchmark categories.
    
    Returns:
        Tuple of (success_rate_df, optimality_df)
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = f"logs/comparison_{timestamp}"
    os.makedirs(base_log_dir, exist_ok=True)
    
    # Results storage: {(model, setting): {category: metrics}}
    results = defaultdict(dict)
    
    # Define all run configurations
    run_configs = []
    
    # Base setting (one-shot with base models)
    for model_name, model_config in BASE_MODELS.items():
        run_configs.append({
            "canonical_name": model_name,
            "setting": "Base",
            "provider": model_config["provider"],
            "model": model_config["model"],
        })
    
    # Thinking setting (one-shot with thinking models)
    for model_name, model_config in THINKING_MODELS.items():
        run_configs.append({
            "canonical_name": model_name,
            "setting": "Thinking",
            "provider": model_config["provider"],
            "model": model_config["model"],
        })
    
    # Omnitune setting (full pipeline with base models)
    for model_name, model_config in BASE_MODELS.items():
        run_configs.append({
            "canonical_name": model_name,
            "setting": "Omnitune",
            "provider": model_config["provider"],
            "model": model_config["model"],
        })
    
    # Random baseline (just one entry, model doesn't matter much)
    run_configs.append({
        "canonical_name": "Random",
        "setting": "Random",
        "provider": BASE_MODELS["ChatGPT"]["provider"],
        "model": BASE_MODELS["ChatGPT"]["model"],
    })
    
    # Run all configurations
    for config in run_configs:
        canonical = config["canonical_name"]
        setting = config["setting"]
        provider = config["provider"]
        model = config["model"]
        
        print(f"\n{'='*60}")
        print(f"Running: {canonical} / {setting}")
        print(f"Provider: {provider}, Model: {model}")
        print(f"{'='*60}")
        
        log_dir = os.path.join(base_log_dir, f"{canonical}_{setting}")
        os.makedirs(log_dir, exist_ok=True)
        
        for category_name, tasks in BENCHMARK_CATEGORIES.items():
            epsilon = EPSILON_MAP[category_name]
            
            category_results = run_benchmark_category(
                category_name=category_name,
                tasks=tasks,
                epsilon=epsilon,
                model_provider=provider,
                model_name=model,
                setting=setting,
                num_iterations=num_iterations,
                log_dir=log_dir,
            )
            
            results[(canonical, setting)][category_name] = category_results
            
            print(f"  {category_name}: success={category_results['success_rate']:.3f}, "
                  f"optimality={category_results['avg_optimality']:.4f}")
    
    # Build DataFrames
    rows_success = []
    rows_optimality = []
    
    for (canonical, setting), category_results in results.items():
        row_success = {"Model": canonical, "Setting": setting}
        row_optimality = {"Model": canonical, "Setting": setting}
        
        for category_name in BENCHMARK_CATEGORIES.keys():
            if category_name in category_results:
                row_success[category_name] = category_results[category_name]["success_rate"]
                opt_val = category_results[category_name]["avg_optimality"]
                row_optimality[category_name] = opt_val if opt_val < float('inf') else None
            else:
                row_success[category_name] = None
                row_optimality[category_name] = None
        
        rows_success.append(row_success)
        rows_optimality.append(row_optimality)
    
    success_df = pd.DataFrame(rows_success)
    optimality_df = pd.DataFrame(rows_optimality)
    
    return success_df, optimality_df


def main():
    print("=" * 80)
    print("MODEL COMPARISON RUNNER")
    print("=" * 80)
    
    # Number of iterations per task
    NUM_ITERATIONS = 5
    
    # Run all experiments
    success_df, optimality_df = run_all_experiments(num_iterations=NUM_ITERATIONS)
    
    # Ensure exports directory exists
    os.makedirs("exports", exist_ok=True)
    
    # Save results
    success_df.to_csv("pass_at_1_results.csv", index=False)
    optimality_df.to_csv("optimality_results.csv", index=False)
    
    print("\n" + "=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)
    
    print("\nSuccess Rate (pass_at_1_results.csv):")
    print(success_df.to_string(index=False))
    
    print("\nOptimality (optimality_results.csv):")
    print(optimality_df.to_string(index=False))


if __name__ == "__main__":
    main()

