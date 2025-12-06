"""
Skyline Ablation
----------------

Utility to evaluate the orchestrator when skyline tracking is disabled
(`use_skyline=False`). For every benchmark/task pair we execute multiple runs of
the orchestrator (default: 5) with T=5 subspace iterations and K=5 refinements.

For each task we persist:
    * success@5 – number of runs that produced a constraint-satisfying solution
    * distance_minus_gt – difference between the best refinement distance and
      the ground-truth distance defined in `scripts/stress_scripts.py`

Results are saved to a CSV for downstream analysis.

Note: Executing this script requires properly configured LLM credentials because
`run_task` will invoke the SubspaceLM/AssignmentLM components.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from config import PROVIDER
from opro.opro_main_loop import run_task
from experiments import bench_marks as bm


# Ground-truth refinement distances copied from `scripts/stress_scripts.py`
GT_TOP_K = [0.36, 0.5, 0.5, 0.03, 0.16, 0.06, 0.08, 0.14]
MAX_DIST_TOP_K = [3, 3, 3, 3, 2, 2, 3, 3]

GT_RANGE = [0.01, 0.02, 0.339, 0.346, 0.11, 0.04, 0.125, 0.1]
MAX_DIST_RANGE = [1, 1, 1, 1, 1, 1, 1, 1]

GT_DIVERSITY = [0.05, 0.08, 0.03333, 0.06, 0.06666667, 0.05, 0.17, 0.25]
MAX_DIST_DIVERSITY = [1, 1, 1, 1, 1, 1, 1, 1]

GT_COMPLEX = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
MAX_DIST_COMPLEX = [2, 2, 2, 2, 3, 4, 4, 4]


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    tasks: Iterable
    gt_distances: Iterable[float]
    max_distances: Iterable[float]
    epsilon: float

    @property
    def task_distance_pairs(self) -> List[Tuple]:
        return list(zip(self.tasks, self.gt_distances, self.max_distances))


BENCHMARKS: Dict[str, BenchmarkConfig] = {
    # "top_k": BenchmarkConfig(
    #     name="top_k",
    #     tasks=[bm.T1a, bm.T1b, bm.T2a, bm.T2b, bm.T3a, bm.T3b, bm.T4a, bm.T4b],
    #     gt_distances=GT_TOP_K,
    #     max_distances=MAX_DIST_TOP_K,
    #     epsilon=0.4,
    # ),
    # "range": BenchmarkConfig(
    #     name="range",
    #     tasks=[bm.T5a, bm.T5b, bm.T6a, bm.T6b, bm.T7a, bm.T7b, bm.T8a, bm.T8b],
    #     gt_distances=GT_RANGE,
    #     max_distances=MAX_DIST_RANGE,
    #     epsilon=0.05,
    # ),
    # "diversity": BenchmarkConfig(
    #     name="diversity",
    #     tasks=[bm.T9a, bm.T9b, bm.T10a, bm.T10b, bm.T11a, bm.T11b, bm.T12a, bm.T12b],
    #     gt_distances=GT_DIVERSITY,
    #     max_distances=MAX_DIST_DIVERSITY,
    #     epsilon=0.0,
    # ),
    "complex": BenchmarkConfig(
        name="complex",
        tasks=[bm.T13a, bm.T13b, bm.T14a, bm.T14b, bm.T15a, bm.T15b, bm.T16a, bm.T16b],
        gt_distances=GT_COMPLEX,
        max_distances=MAX_DIST_COMPLEX,
        epsilon=0.2,
    ),
}


def run_skyline_ablation_test(
    output_csv: str,
    runs_per_task: int = 5,
    max_subspace_iters: int = 5,
    max_refinements_per_subspace: int = 5,
    model_provider: str = PROVIDER,
    model_name: str = None,
) -> pd.DataFrame:
    """
    Execute the skyline-ablation experiment and persist metrics to CSV.

    Args:
        output_csv: Destination path for the results CSV.
        runs_per_task: Number of independent orchestrator runs per task.
        max_subspace_iters: Maximum subspace iterations (T).
        max_refinements_per_subspace: Maximum refinements per subspace (K).

    Returns:
        DataFrame containing the aggregated metrics.
    """

    results: List[Dict[str, object]] = []
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for benchmark_name, config in BENCHMARKS.items():
        for task, gt_distance, max_distance in config.task_distance_pairs:
            success_count = 0
            best_distance = float("inf")
            tokens: List[int] = []

            for run_idx in range(1, runs_per_task + 1):
                log_dir = output_path.parent / "logs" / model_provider / benchmark_name / task.name / f"run_{run_idx}"
                try:
                    refined_query, refinement_distance, token_use = run_task(
                        task=task,
                        epsilon=config.epsilon,
                        perform_analysis=False,
                        perform_optimization=True,
                        max_assignments_per_subspace=max_refinements_per_subspace,
                        max_subspace_iters=max_subspace_iters,
                        assignment_lm_only_mode=False,
                        subspace_lm_only_random=False,
                        random_seed=None,
                        one_shot_mode=False,
                        is_having=False,
                        use_history=False,
                        use_skyline=False,
                        log_dir=str(log_dir),
                        model_provider=model_provider,
                        model_name=model_name,
                    )
                except Exception as exc:
                    refined_query, refinement_distance, token_use = None, float("inf"), 0
                    print(f"[WARN] skyline_ablation {benchmark_name}/{task.name} run {run_idx} failed: {exc}")

                tokens.append(token_use)
                if refined_query is not None and refinement_distance < float("inf"):
                    success_count += 1
                if refinement_distance < best_distance:
                    best_distance = refinement_distance

            best_distance_value = None if best_distance == float("inf") else best_distance
            distance_minus_gt = (
                None if best_distance == float("inf") else best_distance_value - gt_distance  # type: ignore[arg-type]
            )

            results.append(
                {
                    "benchmark": benchmark_name,
                    "task": task.name,
                    "success_at_5": success_count,
                    "best_refinement_distance": best_distance_value,
                    "norm_dist": (max_distance - best_distance) / (max_distance - gt_distance),
                    "tokens_used_mean": sum(tokens) / len(tokens) if tokens else 0,
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run skyline ablation (use_skyline=False).")
    parser.add_argument(
        "--output",
        default="experiments_scripts/exports/skyline_ablation_results_gpt41mini.csv",
        help="Destination CSV path for the aggregated results.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per task (default: 5).")
    parser.add_argument("--max-subspace-iters", type=int, default=5, help="Maximum subspace iterations (T).")
    parser.add_argument(
        "--max-refinements-per-subspace",
        type=int,
        default=5,
        help="Maximum refinements per subspace (K).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MODELS = {
        "openai": "gpt-4.1-mini",
        "google": "gemini-2.0-flash-lite",
        "cloudflare": "@cf/mistralai/mistral-small-3.1-24b-instruct"
    }
    for provider, model in MODELS.items():
        run_skyline_ablation_test(
            output_csv="experiments_scripts/exports/skyline_history_ablation_results_{}.csv".format(provider),
            runs_per_task=1 if provider == "cloudflare" else args.runs,
            max_subspace_iters=args.max_subspace_iters,
            max_refinements_per_subspace=args.max_refinements_per_subspace,
            model_provider=provider,
            model_name=model
        )

