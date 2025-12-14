#!/usr/bin/env python3
# visualize_results.py

"""
Generate visualizations from experiment results.

Usage:
    python visualize_results.py                    # Use most recent results
    python visualize_results.py results/file.csv   # Use specific results file
    python visualize_results.py --show             # Display plots interactively
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.visualization.plot_master import generate_all_plots
from src.results_manager import get_latest_results, get_primary_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate visualizations from sunspot prediction results"
    )
    
    parser.add_argument(
        "csv_file",
        nargs="?",
        help="Path to results CSV file (default: most recent)",
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (default: save only)",
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save plots to disk",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine which results file to use
    if args.csv_file:
        csv_path = Path(args.csv_file)
        if not csv_path.exists():
            print(f"Error: File not found: {csv_path}")
            return 1
    else:
        # Try primary results first, fall back to latest
        csv_path = get_primary_results()
        if csv_path is None:
            csv_path = get_latest_results()
            if csv_path is None:
                print("No results files found in results/ directory")
                print("Run experiments first: python run_experiments.py")
                return 1
        
        # Indicate if using primary results
        if (Path("results") / "experiment_results.csv").exists():
            print("Using PRIMARY results (marked as most important)")
        else:
            print("Using LATEST results")
    
    print("=" * 70)
    print("SUNSPOT PREDICTION VISUALIZATION")
    print("=" * 70)
    print(f"Results file: {csv_path}")
    print()
    
    # Generate plots
    generate_all_plots(
        csv_path,
        save=not args.no_save,
        show=args.show,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
