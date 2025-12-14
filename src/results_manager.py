# src/results_manager.py

"""
Results management for sunspot prediction experiments.

Handles saving experiment results to CSV files for later analysis and visualization.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def save_results_to_csv(
    results: list[dict[str, Any]],
    filename: str | None = None,
    include_timestamp: bool = True,
) -> Path:
    """
    Save experiment results to a CSV file.
    
    Args:
        results: List of result dictionaries from run_experiment()
        filename: Base filename (default: "experiment_results")
        include_timestamp: Add timestamp to filename
    
    Returns:
        Path to saved CSV file
    """
    # Create results directory if it doesn't exist
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Generate filename
    if filename is None:
        filename = "experiment_results"
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}"
    
    if not filename.endswith(".csv"):
        filename = f"{filename}.csv"
    
    filepath = RESULTS_DIR / filename
    
    # Convert results to DataFrame
    # Extract non-nested fields for CSV
    csv_data = []
    for result in results:
        row = {
            "model_name": result["model_name"],
            "start_year": result["start_year"],
            "n_train_samples": result["n_train_samples"],
            "cv_mean_rmse": result["cv_mean_rmse"],
            "test_rmse": result["test_rmse"],
        }
        
        # Add individual fold RMSEs if available (only if CV was run)
        if "cv_fold_rmses" in result and result["cv_fold_rmses"] is not None:
            for i, fold_rmse in enumerate(result["cv_fold_rmses"], 1):
                row[f"cv_fold_{i}_rmse"] = fold_rmse
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    
    return filepath


def save_predictions_to_csv(
    result: dict[str, Any],
    test_dates: pd.DatetimeIndex,
    actual_values: pd.Series,
    filename: str | None = None,
) -> Path:
    """
    Save model predictions along with actual values to CSV.
    
    Args:
        result: Single result dictionary containing predictions
        test_dates: DatetimeIndex for test period
        actual_values: Actual SSN values for test period
        filename: Base filename (default: model_name_startyear_predictions)
    
    Returns:
        Path to saved CSV file
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Generate filename
    if filename is None:
        filename = f"{result['model_name']}_y{result['start_year']}_predictions"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.csv"
    filepath = RESULTS_DIR / filename
    
    # Create DataFrame with dates, actual, and predicted
    df = pd.DataFrame({
        "date": test_dates,
        "actual_ssn": actual_values,
        "predicted_ssn": result["predictions"],
        "error": actual_values - result["predictions"],
        "absolute_error": abs(actual_values - result["predictions"]),
    })
    
    # Add metadata as comments would be lost in CSV, so add as first row then skip
    # Instead, just save clean data
    df.to_csv(filepath, index=False)
    
    return filepath


def load_results_from_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Load experiment results from a CSV file.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        DataFrame with experiment results
    """
    return pd.read_csv(filepath)


def get_latest_results(pattern: str = "experiment_results_*.csv") -> Path | None:
    """
    Get the most recently created results file matching a pattern.
    
    Args:
        pattern: Glob pattern for matching files
    
    Returns:
        Path to most recent file, or None if no files found
    """
    files = list(RESULTS_DIR.glob(pattern))
    if not files:
        return None
    
    # Sort by modification time, return most recent
    return max(files, key=lambda p: p.stat().st_mtime)


def get_primary_results() -> Path | None:
    """
    Get the primary (marked as most important) results file.
    
    Checks for:
    1. results/experiment_results.csv (primary results copy)
    2. Falls back to most recent results
    
    Returns:
        Path to primary results file, or None if no results found
    """
    primary_path = RESULTS_DIR / "experiment_results.csv"
    
    if primary_path.exists():
        return primary_path
    
    # Fall back to latest results
    return get_latest_results()


def mark_as_primary(csv_path: str | Path) -> None:
    """
    Mark a results file as the primary one to use for visualizations.
    
    This copies the specified file to results/experiment_results.csv.
    Use this for your dense/complete experiment results that you want to reuse.
    
    Args:
        csv_path: Path to the results file to mark as primary
    """
    import shutil
    
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    primary_copy = RESULTS_DIR / "experiment_results.csv"
    
    # Copy the file
    shutil.copy2(csv_path, primary_copy)
    
    print(f"Marked as primary: {csv_path}")
    print(f"Copied to: {primary_copy}")
