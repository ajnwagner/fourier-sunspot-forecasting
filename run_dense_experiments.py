#!/usr/bin/env python3
# run_dense_experiments.py

"""
[DEPRECATED] Use main.py instead with --dense flag:
    python main.py --dense 1900 1920

This script is kept for backwards compatibility but main.py now handles
both sparse and dense year sampling.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.experiments.run_sunspot_experiments import run_experiment
from src.models.registry import AVAILABLE_MODELS, DEFAULT_MODELS
from src.results_manager import mark_as_primary, save_results_to_csv

print("WARNING: run_dense_experiments.py is deprecated.")
print("Use instead: python main.py --dense <min_year> <max_year>")
print("Continuing anyway...\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run dense sunspot prediction experiments (every year)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=AVAILABLE_MODELS,
        help="Models to test (default: all)",
    )
    
    parser.add_argument(
        "--start-year-min",
        type=int,
        default=1749,
        help="First start year to test (default: 1749)",
    )
    
    parser.add_argument(
        "--start-year-max",
        type=int,
        default=1940,
        help="Last start year to test (default: 1940)",
    )
    
    parser.add_argument(
        "--year-step",
        type=int,
        default=1,
        help="Step between years (default: 1 for dense sampling, use 5 or 10 for faster)",
    )
    
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Perform cross-validation (slower, but provides CV RMSE). Default: off",
    )
    
    parser.add_argument(
        "--mark-primary",
        action="store_true",
        help="Mark these results as PRIMARY for reuse in visualizations",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine which models to run
    if args.models:
        model_names = args.models
    else:
        # Default: all models for publication-quality plots
        model_names = AVAILABLE_MODELS
    
    # Generate dense year range
    start_years = list(range(args.start_year_min, args.start_year_max + 1, args.year_step))
    
    verbose = not args.quiet
    
    if verbose:
        print("=" * 70)
        print("DENSE SUNSPOT PREDICTION EXPERIMENTS")
        print("=" * 70)
        print(f"Models: {', '.join(model_names)}")
        print(f"Start years: {args.start_year_min} to {args.start_year_max} (step={args.year_step})")
        print(f"Total years: {len(start_years)}")
        print(f"Total configurations: {len(model_names) * len(start_years)}")
        print("\nThis may take a while...")
        print("=" * 70)
        print()
    
    # Run experiments
    results, best = run_experiment(
        model_names=model_names,
        start_years=start_years,
        verbose=verbose,
        save_to_csv=True,
        save_best_predictions=True,
        skip_cv=not args.cv,  # Skip CV unless --cv flag is set
    )
    
    # Mark as primary if requested
    if args.mark_primary and verbose:
        # Get the most recent results file (the one we just created)
        from src.results_manager import get_latest_results
        latest_csv = get_latest_results()
        if latest_csv:
            mark_as_primary(latest_csv)
            print(f"\n✓ Marked as PRIMARY results for future visualizations")
            print(f"  Use 'python visualize_results.py' to visualize this data")
    
    if verbose:
        # Print summary statistics
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        # Group by model
        for model_name in model_names:
            model_results = [r for r in results if r["model_name"] == model_name]
            test_rmses = [r["test_rmse"] for r in model_results]
            
            # Find minimum RMSE and corresponding year
            min_idx = test_rmses.index(min(test_rmses))
            min_year = model_results[min_idx]["start_year"]
            
            print(f"\n{model_name}:")
            print(f"  Test RMSE: min={min(test_rmses):.2f} (year {min_year}), "
                  f"max={max(test_rmses):.2f}, mean={sum(test_rmses)/len(test_rmses):.2f}")
        
        print("\n" + "=" * 70)
        print("NEXT STEP: Generate plots with:")
        print("  python visualize_results.py")
        print("=" * 70)
    
    return results, best


if __name__ == "__main__":
    main()
