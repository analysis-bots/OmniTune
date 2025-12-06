"""
Shared constants, configurations, and helper functions for experiments.
"""

import os
import sys
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.bench_marks import (
    T1a, T1b, T2a, T2b, T3a, T3b, T4a, T4b,  # Top-K benchmarks
    T5a, T5b, T6a, T6b, T7a, T7b, T8a, T8b,  # Range query benchmarks
    T9a, T9b, T10a, T10b, T11a, T11b, T12a, T12b,  # Diversity benchmarks
    T13a, T13b, T14a, T14b, T15a, T15b, T16a, T16b,  # Complex benchmarks
)
from opro.opro_main_loop import run_task

# ============================================================================
# Benchmark Categories
# ============================================================================

BENCHMARK_CATEGORIES = {
    "top_k": {
        "tasks": [T1a, T1b, T2a, T2b, T3a, T3b, T4a, T4b],
        "epsilon": 0.4,
        "gt": pd.Series([0.36, 0.5, 0.5, 0.03, 0.16, 0.06, 0.08, 0.14]),
        "max_dist": pd.Series([3, 3, 3, 3, 2, 2, 3, 3]),
    },
    "range": {
        "tasks": [T5a, T5b, T6a, T6b, T7a, T7b, T8a, T8b],
        "epsilon": 0.05,
        "gt": pd.Series([0.01, 0.02, 0.503, 0.346, 0.11, 0.04, 0.125, 0.1]),
        "max_dist": pd.Series([1, 1, 1, 1, 1, 1, 1, 1]),
    },
    "diversity": {
        "tasks": [T9a, T9b, T10a, T10b, T11a, T11b, T12a, T12b],
        "epsilon": 0.0,
        "gt": pd.Series([0.05, 0.08, 0.03333, 0.06, 0.06666667, 0.05, 0.17, 0.25]),
        "max_dist": pd.Series([1, 1, 1, 1, 1, 1, 1, 1]),
    },
    "complex": {
        "tasks": [T13a, T13b, T14a, T14b, T15a, T15b, T16a, T16b],
        "epsilon": 0.2,
        "gt": pd.Series([0, 0, 0, 0, 0, 0, 0, 0]),
        "max_dist": pd.Series([2, 2, 2, 2, 3, 4, 4, 4]),
    },
}

# ============================================================================
# Model Configurations
# ============================================================================

BASE_MODELS = {
    "chatgpt": {"provider": "openai", "model": "gpt-4.1-mini"},
    "gemini": {"provider": "google", "model": "gemini-2.0-flash-lite"},
    "mistral": {"provider": "cloudflare", "model": "@cf/mistralai/mistral-small-3.1-24b-instruct"},
}

THINKING_MODELS = {
    "chatgpt": {"provider": "openai", "model": "gpt-5-mini"},
    "gemini": {"provider": "google", "model": "gemini-2.5-flash-lite"},
    "mistral": {"provider": "mistral", "model": "magistral-small-latest"},
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_benchmarks(benchmark_names: List[str]) -> Dict[str, Dict]:
    """Get benchmark configurations for the specified categories."""
    if "all" in benchmark_names:
        return BENCHMARK_CATEGORIES
    return {name: BENCHMARK_CATEGORIES[name] for name in benchmark_names if name in BENCHMARK_CATEGORIES}


def run_single_task(
    task,
    epsilon: float,
    model_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    one_shot_mode: bool = False,
    assignment_lm_only: bool = False,
    subspace_lm_only_random: bool = False,
    log_dir: str = "logs",
    is_having: bool = False,
    max_subspace_iters: int = 5,
    max_assignments_per_subspace: int = 5,
    seed: int = 42,
    verbose: bool = False,
    use_history: bool = True,
    use_skyline: bool = True,
) -> Tuple[Optional[str], float, int, bool]:
    """Run a single task and return results."""
    try:
        if verbose:
            print(f"      -> Starting task execution...")
            print(f"         epsilon={epsilon}, T={max_subspace_iters}, K={max_assignments_per_subspace}")
        
        refined_query, refinement_distance, token_use = run_task(
            task,
            epsilon,
            perform_analysis=False,
            one_shot_mode=one_shot_mode,
            assignment_lm_only_mode=assignment_lm_only,
            subspace_lm_only_random=subspace_lm_only_random,
            is_having=is_having,
            log_dir=log_dir,
            random_seed=seed,
            max_assignments_per_subspace=max_assignments_per_subspace,
            max_subspace_iters=max_subspace_iters,
            model_provider=model_provider,
            model_name=model_name,
            use_history=use_history,
            use_skyline=use_skyline,
        )
        is_successful = refinement_distance < float('inf')
        
        if verbose:
            status = "SUCCESS" if is_successful else "FAILED"
            print(f"      -> Result: {status}, dist={refinement_distance:.4f}, tokens={token_use}")
        
        return refined_query, refinement_distance, token_use, is_successful
    except Exception as e:
        print(f"      -> ERROR: {e}")
        return None, float('inf'), 0, False


def compute_optimality(best_dist: float, max_dist: float, gt: float) -> float:
    """Compute optimality score using normalized distance formula."""
    if best_dist == float('inf'):
        return 0.0
    return (max_dist - best_dist) / (max_dist - gt + 1e-9)


