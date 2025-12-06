"""
Ablation Study Experiment

Compares different configurations of OmniTune:
- omnitune: Full OmniTune (Actor + Critic + History + Skyline)
- assignment_lm_only: Only Assignment LM (no Subspace LM)
- subspace_lm_only: Only Subspace LM (random assignments)
- no_history: No history context, with skyline
- no_history_no_skyline: No history, no skyline
"""

import os
import datetime

import pandas as pd

from experiments.common import (
    get_benchmarks,
    run_single_task,
    BASE_MODELS,
)


# Ablation configurations: (name, assignment_lm_only, subspace_lm_only_random, use_history, use_skyline)
ABLATION_CONFIGS = [
    ("omnitune", False, False, True, True),              # Full Omnitune
    ("assignment_lm_only", True, False, True, True),     # Assignment LM only
    ("subspace_lm_only", False, True, True, True),       # Subspace LM + random assignments
    ("no_history", False, False, False, True),           # No history, with skyline
    ("no_history_no_skyline", False, False, False, False),  # No history, no skyline
]


def run_ablation(args):
    """Run ablation study comparing different configurations."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get model configuration
    model_key = args.model.lower()
    model_config = BASE_MODELS.get(model_key, BASE_MODELS["chatgpt"])
    model_provider = model_config["provider"]
    model_name = model_config["model"]
    
    run_name = f"ablation_{model_key}_{timestamp}"
    benchmarks = get_benchmarks(args.benchmarks)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_configs = len(benchmarks) * len(ABLATION_CONFIGS)
    config_idx = 0
    
    print(f"\n[ABLATION STUDY]")
    print(f"  Run name: {run_name}")
    print(f"  Model: {model_key} ({model_provider}: {model_name})")
    print(f"  Benchmarks: {list(benchmarks.keys())}")
    print(f"  Configurations: {[c[0] for c in ABLATION_CONFIGS]}")
    print(f"  Total configurations: {total_configs}")
    print(f"  Iterations per task: {args.iterations}")
    
    for bench_name, bench_config in benchmarks.items():
        tasks = bench_config["tasks"]
        epsilon = bench_config["epsilon"]
        gt = bench_config["gt"]
        max_dist = bench_config["max_dist"]
        is_having = (bench_name == "complex")
        
        for config_name, assignment_lm_only, subspace_lm_only_random, use_history, use_skyline in ABLATION_CONFIGS:
            config_idx += 1
            out_path = os.path.join(args.output_dir, f"{run_name}_{bench_name}_{config_name}.csv")
            log_dir = f"logs/{run_name}_{bench_name}_{config_name}"
            
            print(f"\n{'='*60}")
            print(f"[{config_idx}/{total_configs}] Ablation: {bench_name} / {config_name}")
            print(f"{'='*60}")
            print(f"  Tasks: {len(tasks)}")
            print(f"  Epsilon: {epsilon}")
            print(f"  Assignment LM only: {assignment_lm_only}")
            print(f"  Subspace LM random: {subspace_lm_only_random}")
            print(f"  Use history: {use_history}")
            print(f"  Use skyline: {use_skyline}")
            print(f"  Output: {out_path}")
            print()
            
            results = {
                'task_name': [], 'original_query': [], 'refined_query': [],
                'refinement_distance': [], 'num_successful': [], 'token_use': [],
                'success_rate': [], 'optimality': []
            }
            
            for task_idx, task in enumerate(tasks):
                best_query, best_dist = None, float('inf')
                total_tokens, num_success = 0, 0
                
                print(f"  [{task_idx+1}/{len(tasks)}] Task: {task.name}")
                
                for i in range(args.iterations):
                    print(f"    Iteration {i+1}/{args.iterations}...", end=" ", flush=True)
                    query, dist, tokens, success = run_single_task(
                        task, epsilon,
                        model_provider=model_provider,
                        model_name=model_name,
                        assignment_lm_only=assignment_lm_only,
                        subspace_lm_only_random=subspace_lm_only_random,
                        is_having=is_having,
                        log_dir=log_dir,
                        seed=args.seed + i,
                        use_history=use_history,
                        use_skyline=use_skyline,
                    )
                    total_tokens += tokens
                    status = "OK" if success else "FAIL"
                    print(f"{status} (dist={dist:.4f}, tokens={tokens})")
                    
                    if success:
                        num_success += 1
                    if dist < best_dist:
                        best_dist = dist
                        best_query = query
                
                # Compute success rate and optimality
                success_rate = num_success / args.iterations
                if best_dist == float('inf'):
                    optimality = 0.0
                else:
                    optimality = (max_dist.iloc[task_idx] - best_dist) / (max_dist.iloc[task_idx] - gt.iloc[task_idx] + 1e-9)
                
                print(f"    -> Summary: {num_success}/{args.iterations} ({success_rate*100:.1f}%), best_dist={best_dist:.4f}, optimality={optimality:.4f}\n")
                
                results['task_name'].append(task.name)
                results['original_query'].append(task.original_query.strip())
                results['refined_query'].append(best_query)
                results['refinement_distance'].append(best_dist)
                results['num_successful'].append(num_success)
                results['token_use'].append(total_tokens / args.iterations)
                results['success_rate'].append(success_rate)
                results['optimality'].append(optimality)
            
            df = pd.DataFrame(results)
            df.to_csv(out_path, index=False)
            
            # Print summary for this config
            avg_success = df['success_rate'].mean()
            avg_optimality = df['optimality'].mean()
            print(f"  [SUMMARY] Avg Success: {avg_success*100:.1f}%, Avg Optimality: {avg_optimality:.4f}")
            print(f"  [SAVED] {out_path}")


