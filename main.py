#!/usr/bin/env python3
"""
OmniTune Experiment CLI

Unified entry point to run different experiment types:
- ablation: Compare different OmniTune configurations
- comparison: Compare LLM providers (ChatGPT/Gemini/Mistral) with Base/Thinking/Omnitune settings
- parameters: Grid search over T (subspaces) and K (refinements per subspace)

Usage:
    python main.py --experiment ablation --model chatgpt --benchmarks top_k range --iterations 5
    python main.py --experiment comparison --benchmarks all --iterations 5 --plot
    python main.py --experiment parameters --benchmarks all --iterations 5 --plot
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Unified Experiment CLI for OmniTune",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --experiment ablation --model chatgpt --benchmarks top_k range --iterations 5
  python main.py --experiment comparison --iterations 5 --plot
  python main.py --experiment parameters --benchmarks all --iterations 5 --plot
        """
    )
    
    parser.add_argument(
        "--experiment", "-e",
        required=True,
        choices=["ablation", "comparison", "parameters"],
        help="Type of experiment to run"
    )
    parser.add_argument(
        "--benchmarks", "-b",
        nargs="+",
        default=["all"],
        choices=["top_k", "range", "diversity", "complex", "all"],
        help="Benchmark categories to run (default: all)"
    )
    parser.add_argument(
        "--model", "-m",
        default="chatgpt",
        choices=["chatgpt", "gemini", "mistral"],
        help="Model family for ablation experiment (default: chatgpt)"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=5,
        help="Number of iterations per task (default: 5)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="exports",
        help="Output directory for results (default: exports)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--subspaces",
        default="1,3,5,7,10",
        help="Comma-separated T values for parameters experiment (default: 1,3,5,7,10)"
    )
    parser.add_argument(
        "--refinements",
        default="1,3,5,7,10",
        help="Comma-separated K values for parameters experiment (default: 1,3,5,7,10)"
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Generate plots after running experiment (comparison and parameters only)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"OMNITUNE EXPERIMENT: {args.experiment.upper()}")
    print("=" * 80)
    print(f"  Benchmarks: {args.benchmarks}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Output: {args.output_dir}")
    print(f"  Seed: {args.seed}")
    if args.experiment == "ablation":
        print(f"  Model: {args.model}")
    if args.experiment == "parameters":
        print(f"  Subspaces (T): {args.subspaces}")
        print(f"  Refinements (K): {args.refinements}")
    print(f"  Plot: {args.plot}")
    print("=" * 80)
    
    # Import and run the appropriate experiment
    if args.experiment == "ablation":
        from experiments.ablation import run_ablation
        run_ablation(args)
    elif args.experiment == "comparison":
        from experiments.comparison import run_comparison
        run_comparison(args)
    elif args.experiment == "parameters":
        from experiments.parameters import run_parameters
        run_parameters(args)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()


