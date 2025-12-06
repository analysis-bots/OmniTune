import csv
import datetime
import os
import traceback
from typing import List, Dict, Any

# Assuming analysis_based_new is in the parent directory relative to experiments
# Adjust import path if needed
import sys

from config import LOG_DIR
from functionality.task import Task

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from opro.opro_main_loop import OmniTuneEngine

# Import benchmark tasks and base Task class
from experiments.bench_marks import (
    T1a, T1b, T2a, T2b, T3a, T3b, T4a, T4b, # Top K
    T5a, T5b, T6a, T6b, T7a, T7b, T8a, T8b, # Range Query
    T9a, T9b, T10a, T10b, T11a, T11b, T12a, T12b # Diversity
)
# Distance functions are now part of the Task object, no need to import them here
# from functionality.objectives import (...)

# --- Configuration ---
MAX_SUBSPACE_ITERS = 2
MAX_REFINEMENTS_PER_SUBSPACE = 3
RESULTS_CSV_FILE = "exports/benchmark_results_llama3_1.csv"
SUMMARY_LOG_FILE = "logs/run_summary_llama3_1.log"
# ---------------------

# Define the list of benchmarks to run
benchmarks_to_run: List[Task] = [
    T1a, T1b, T2a, T2b, T3a, T3b, T4a, T4b, # Top K Tasks
    T5a, T5b, T6a, T6b, T7a, T7b, T8a, T8b, # Range Query Tasks
    T9a, T9b, T10a, T10b, T11a, T11b, T12a, T12b # Add Diversity Tasks here if needed
]

def get_orchestrator_inputs(task: Task) -> Dict[str, Any]:
    """Generates string inputs and distance function required by OmniTuneEngine from a Task object."""

    # 1. Format Constraints (Using existing task attribute)
    constraints_str = task.constraints_str

    # 2. Get Refinement Distance Function and Metric String (From task attributes)
    distance_func = task.refinement_objective
    refinement_distance_metric_str = task.refinement_objective_str
    if not refinement_distance_metric_str:
        refinement_distance_metric_str = "(No description provided in Task object)"

    # 3. Define Refineable Predicates (Using existing task attribute)
    refineable_predicates_str = task.alterable_attributes_str
    if not refineable_predicates_str:
        refineable_predicates_str = "(No alterable attributes defined in Task object)"

    epsilon = task.epsilons[-1] # Assuming last epsilon is the one to use

    # Validate that distance_func is callable
    if not callable(distance_func):
        print(f"Warning: task.refinement_objective for task '{task.name}' is not callable. Using dummy function.")
        distance_func = lambda refined_query: 999.0
        # Optionally update the description string too
        # refinement_distance_metric_str += " [Warning: Objective not callable!]"

    return {
        "constraints": constraints_str,
        "epsilon": epsilon,
        "refinement_distance_metric": refinement_distance_metric_str, # String description for LLM
        "distance_func": distance_func, # Actual function for calculation
        "refineable_predicates": refineable_predicates_str,
    }

def main():
    print("Starting Benchmark Run...")
    start_time = datetime.datetime.now()

    results_data = []
    csv_headers = [
        "run_id", "task_name", "task_type", "timestamp", "original_query",
        "best_query", "refinement_distance", "status", "error_message", "log_directory"
    ]

    # Ensure logs directory exists (orchestrator also creates subdirs)
    if not os.path.exists("logs/{LOG_FOLDER}"):
        os.makedirs("logs/{LOG_FOLDER}")

    # Open summary log for appending
    with open(SUMMARY_LOG_FILE, "a", encoding="utf-8") as slog:
        slog.write(f"\n=== Benchmark Run Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

        for i, task in enumerate(benchmarks_to_run):
            run_timestamp = datetime.datetime.now()
            run_id = f"{task.name}_{run_timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
            print(f"\n--- Running Task {i+1}/{len(benchmarks_to_run)}: {run_id} ---")
            slog.write(f"\n* Task: {run_id} (Type: {task.task_type.name})\n")
            slog.write(f"  Original Query:\n{task.original_query}\n")

            status = "Failure"
            error_msg = "N/A"
            best_query_res = "N/A"
            best_dist_res = float('inf')
            log_dir = f"{LOG_DIR}/{run_id}" # Expected log dir path

            try:
                # Get inputs for the orchestrator
                orchestrator_inputs = get_orchestrator_inputs(task)

                print(f"  Refineable Predicates:\n{orchestrator_inputs['refineable_predicates']}")
                print(f"  Constraints:\n{orchestrator_inputs['constraints']}")
                print(f"  Distance Metric Desc: {orchestrator_inputs['refinement_distance_metric']}")

                # Check if dataframe is empty before proceeding
                if task.df.empty:
                    raise ValueError("Input DataFrame is empty.")

                # Check if distance function was resolved (added validation in helper)
                # if orchestrator_inputs["distance_func"] is None:
                #     raise ValueError(f"Distance function could not be determined for task type {task.task_type.name}")

                # Instantiate and run orchestrator
                orchestrator = OmniTuneEngine(
                    task_name=run_id, # Use run_id for unique logging
                    original_query=task.original_query,
                    input_dataset=task.df,
                    constraints=orchestrator_inputs["constraints"],
                    epsilon=orchestrator_inputs["epsilon"],
                    refinement_distance_metric=orchestrator_inputs["refinement_distance_metric"], # Pass string desc
                    distance_func=orchestrator_inputs["distance_func"], # Pass actual function
                    refineable_predicates_str=orchestrator_inputs["refineable_predicates"],
                    max_subspace_iters=MAX_SUBSPACE_ITERS,
                    max_refinements=MAX_REFINEMENTS_PER_SUBSPACE
                )

                best_query_res, best_dist_res = orchestrator.run_refinement_loop()

                status = "Success"
                if best_query_res is None:
                    status = "Completed (No Solution Found)"
                    best_query_res = "N/A"
                    best_dist_res = float('inf')
                    slog.write(f"  Outcome: Completed, but no satisfying query found.\n")
                else:
                     slog.write(f"  Outcome: Success!\n")
                     slog.write(f"  Best Query:\n{best_query_res}\n")
                     slog.write(f"  Best Distance: {best_dist_res}\n")

            except Exception as e:
                print(f"ERROR running task {run_id}: {e}")
                traceback.print_exc() # Print full traceback for debugging
                error_msg = str(e)
                status = "Failure"
                slog.write(f"  Outcome: FAILED!\n")
                slog.write(f"  Error: {error_msg}\n")
                slog.write(f"{traceback.format_exc()}\n") # Log traceback too
            
            finally:
                 # Append results to list for CSV
                results_data.append({
                    "run_id": run_id,
                    "task_name": task.name,
                    "task_type": task.task_type.name,
                    "timestamp": run_timestamp.isoformat(),
                    "original_query": task.original_query,
                    "best_query": best_query_res,
                    "refinement_distance": best_dist_res if best_dist_res != float('inf') else 'inf',
                    "status": status,
                    "error_message": error_msg,
                    "log_directory": log_dir
                })
                print(f"--- Finished Task {run_id} with status: {status} ---")

    # Write results to CSV
    print(f"\nWriting results to {RESULTS_CSV_FILE}...")
    try:
        with open(RESULTS_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(results_data)
        print("CSV write successful.")
    except IOError as e:
        print(f"Error writing CSV file: {e}")

    end_time = datetime.datetime.now()
    print(f"\nBenchmark Run Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {end_time - start_time}")

if __name__ == "__main__":
    main() 
