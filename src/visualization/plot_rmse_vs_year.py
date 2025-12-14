# src/visualization/plot_rmse_vs_year.py

"""
Plot RMSE vs. training start year for multiple algorithms.
Replicates the dense, smooth plots from the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "rmse_analysis"


def plot_rmse_vs_year(
    csv_path: str | Path,
    metric: str = "test_rmse",
    last_n_years: int | None = None,
    save: bool = True,
    show: bool = False,
    output_filename: str | None = None,
) -> None:
    """
    Plot RMSE vs. training start year for multiple algorithms.
    
    Creates dense, smooth plots with all available data points.
    
    Args:
        csv_path: Path to experiment results CSV
        metric: Metric to plot ("test_rmse" or "cv_mean_rmse")
        last_n_years: If specified, only plot last N years of start years
        save: Save figure to disk
        show: Display figure interactively
        output_filename: Custom output filename (default: auto-generated)
    """
    FIGURES_DIR.mkdir(exist_ok=True)
    
    # Load results
    df = pd.read_csv(csv_path)
    
    # Filter to last N years if specified
    if last_n_years is not None:
        max_year = df["start_year"].max()
        min_year = max_year - last_n_years
        df = df[df["start_year"] >= min_year]
    
    # Get unique models
    models = sorted(df["model_name"].unique())
    
    # Create figure with larger size for better visibility
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Color palette for models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Track all RMSE values to find global minimum (only for test RMSE)
    global_min_rmse = float('inf')
    global_min_year = None
    global_min_model = None
    
    # Plot each model with smooth lines (NO markers)
    for idx, model in enumerate(models):
        model_data = df[df["model_name"] == model].sort_values("start_year")
        
        # Check if this model has the global minimum (only for test RMSE)
        if metric == "test_rmse":
            model_min_idx = model_data[metric].idxmin()
            model_min_rmse = model_data.loc[model_min_idx, metric]
            if model_min_rmse < global_min_rmse:
                global_min_rmse = model_min_rmse
                global_min_year = model_data.loc[model_min_idx, "start_year"]
                global_min_model = model
        
        # Create smooth line WITHOUT markers
        ax.plot(
            model_data["start_year"],
            model_data[metric],
            label=model.replace("_", " ").title(),
            linewidth=2.5,
            color=colors[idx % len(colors)],
            alpha=0.85,
        )
    
    # Plot ONLY the global minimum point with black X marker (only for test RMSE)
    if metric == "test_rmse" and global_min_year is not None and global_min_model is not None:
        ax.plot(
            global_min_year,
            global_min_rmse,
            marker='x',
            markersize=12,
            markeredgewidth=3,
            color='black',
            zorder=10,
        )
        
        # Add label below the X
        ax.text(
            global_min_year,
            global_min_rmse - (ax.get_ylim()[1] * 0.04),  # Slightly below the point
            f"{global_min_model.replace('_', ' ').title()}\n{int(global_min_year)}, {global_min_rmse:.2f}",
            ha='center',
            va='top',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8)
        )
    
    # Labels and styling - different for CV vs test RMSE
    if metric == "cv_mean_rmse":
        metric_name = "Cross-Validation RMSE"
        title_prefix = "Cross-Validation RMSE"
        y_range = (40, 90)
    else:
        metric_name = "Test RMSE"
        title_prefix = "Test RMSE"
        y_range = (0, None)
    
    ax.set_xlabel("Year of Time Series Start", fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
    
    # Set y-axis range based on metric type
    ax.set_ylim(bottom=y_range[0], top=y_range[1])
    
    # Set x-axis to use integer years only
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto'))
    
    # Grid for easier reading
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Legend with better positioning
    ax.legend(fontsize=12, framealpha=0.95, loc='best', edgecolor='black')
    
    # Title
    title = f"{title_prefix} vs. Training Start Year"
    if last_n_years is not None:
        title += f" (Last {last_n_years} Years)"
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    
    # Tighter layout
    plt.tight_layout()
    
    if save:
        if output_filename is None:
            # Include year range in filename
            min_year = int(df["start_year"].min())
            max_year = int(df["start_year"].max())
            metric_suffix = "test" if metric == "test_rmse" else "cv"
            output_filename = f"rmse_vs_year_{metric_suffix}_{min_year}_{max_year}.png"
        output_path = FIGURES_DIR / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    import argparse
    from src.results_manager import get_latest_results
    
    parser = argparse.ArgumentParser(description="Plot RMSE vs training start year")
    parser.add_argument("csv_path", nargs="?", help="Path to experiment results CSV (default: latest)")
    parser.add_argument("--metric", choices=["test", "cv"], default="test",
                        help="RMSE metric to plot: 'test' for test_rmse or 'cv' for cv_mean_rmse (default: test)")
    parser.add_argument("--start-year", type=int, help="Minimum training start year to include")
    parser.add_argument("--end-year", type=int, help="Maximum training start year to include")
    parser.add_argument("--output", help="Custom output filename")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    
    args = parser.parse_args()
    
    # Get CSV path
    csv_path = args.csv_path
    if csv_path is None:
        csv_path = get_latest_results()
        if csv_path is None:
            print("No results found. Run experiments first.")
            exit(1)
    
    # Read data to filter by year range if needed
    df = pd.read_csv(csv_path)
    if args.start_year is not None:
        df = df[df["start_year"] >= args.start_year]
    if args.end_year is not None:
        df = df[df["start_year"] <= args.end_year]
    
    # Save filtered data to temp file if filtering was applied
    if args.start_year is not None or args.end_year is not None:
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        csv_path = temp_file.name
    
    # Map metric argument to column name
    metric_col = "test_rmse" if args.metric == "test" else "cv_mean_rmse"
    
    # Plot
    plot_rmse_vs_year(
        csv_path,
        metric=metric_col,
        save=True,
        show=args.show,
        output_filename=args.output
    )

