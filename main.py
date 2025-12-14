#!/usr/bin/env python3
# main.py

"""
Main entry point for sunspot prediction experiments.

Usage:
    python main.py                              # Quick test (2 models, 5 years)
    python main.py --all-models                 # Test all models (including slow GP)
    python main.py --dense 1900 1920            # Dense year sampling for plots
    python main.py --full                       # Full sweep: all models, 1749-1940
    
Note:
    - Default excludes gaussian_process (very slow)
    - Cross-validation is OFF by default (use --cv to enable)
    - Use --dense for publication-quality RMSE plots
"""

import argparse
import sys
import warnings

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.experiments.run_sunspot_experiments import run_experiment
from src.models.registry import AVAILABLE_MODELS, DEFAULT_MODELS
from src.results_manager import mark_as_primary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sunspot prediction experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Quick test
  python main.py --dense 1900 1920           # Dense sampling for RMSE plot
  python main.py --all-models --cv           # All models with cross-validation
  python main.py --full --mark-primary       # Full run, mark as primary
        """
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=AVAILABLE_MODELS,
        help="Models to test (default: all except gaussian_process)",
    )
    
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Test ALL models including gaussian_process (slow)",
    )
    
    parser.add_argument(
        "--dense",
        nargs=2,
        type=int,
        metavar=("MIN_YEAR", "MAX_YEAR"),
        help="Dense year sampling: test EVERY year from MIN_YEAR to MAX_YEAR",
    )
    
    parser.add_argument(
        "--start-years",
        nargs="+",
        type=int,
        help="Specific start years to test (default: 1749, 1800, 1850, 1900, 1940)",
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full sweep: default models, years 1749-1940 in steps of 10",
    )
    
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Enable cross-validation (slower but more rigorous)",
    )
    
    parser.add_argument(
        "--merit-weights",
        action="store_true",
        help="Apply Merit Attribute weighting to lag features (as per original paper)",
    )
    
    parser.add_argument(
        "--mark-primary",
        action="store_true",
        help="Mark results as PRIMARY for reuse in visualizations",
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
    if args.all_models:
        model_names = AVAILABLE_MODELS
    elif args.models:
        model_names = args.models
    else:
        # Default: all models except gaussian_process
        model_names = DEFAULT_MODELS
    
    # Determine which start years to test
    if args.dense:
        # Dense sampling: every year in range
        min_year, max_year = args.dense
        start_years = list(range(min_year, max_year + 1))
    elif args.full:
        start_years = list(range(1749, 1941, 10))
    elif args.start_years:
        start_years = args.start_years
    else:
        # Default: selected years for quick test
        start_years = [1749, 1800, 1850, 1900, 1940]
    
    # Run experiments
    verbose = not args.quiet
    
    if verbose:
        print("=" * 70)
        print("SUNSPOT PREDICTION EXPERIMENTS")
        print("=" * 70)
        print(f"Models: {', '.join(model_names)}")
        print(f"Start years: {min(start_years)} to {max(start_years)} ({len(start_years)} values)")
        print(f"Total configurations: {len(model_names) * len(start_years)}")
        print(f"Cross-validation: {'ON' if args.cv else 'OFF'}")
        print(f"Merit Attribute weighting: {'ON' if args.merit_weights else 'OFF'}")
        if args.dense:
            print("Mode: DENSE sampling (every year)")
        print("=" * 70)
        print()
    
    results, best = run_experiment(
        model_names=model_names,
        start_years=start_years,
        verbose=verbose,
        save_to_csv=True,
        save_best_predictions=True,
        skip_cv=not args.cv,
        apply_merit_weights=args.merit_weights,
    )
    
    # Mark as primary if requested
    if args.mark_primary:
        from src.results_manager import get_latest_results
        latest_csv = get_latest_results()
        if latest_csv:
            mark_as_primary(latest_csv)
            if verbose:
                print(f"\n✓ Marked as PRIMARY results for future visualizations")
    
    if verbose:
        # Print summary statistics
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        # Group by model
        for model_name in model_names:
            model_results = [r for r in results if r["model_name"] == model_name]
            test_rmses = [r["test_rmse"] for r in model_results]
            
            print(f"\n{model_name}:")
            if args.cv and model_results[0]["cv_mean_rmse"] is not None:
                cv_rmses = [r["cv_mean_rmse"] for r in model_results]
                print(f"  CV RMSE:   min={min(cv_rmses):.2f}, max={max(cv_rmses):.2f}, mean={sum(cv_rmses)/len(cv_rmses):.2f}")
            print(f"  Test RMSE: min={min(test_rmses):.2f} (year {model_results[test_rmses.index(min(test_rmses))]['start_year']}), max={max(test_rmses):.2f}, mean={sum(test_rmses)/len(test_rmses):.2f}")
    
    return results, best


if __name__ == "__main__":
    main()
