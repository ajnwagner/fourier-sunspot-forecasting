# src/experiments/run_sunspot_experiments.py

"""
Sunspot prediction experiments using time-series cross-validation.

This module implements the experiment protocol from the paper:
- Fixed 131-month test window
- Variable training start years (1749-1940)
- 6-fold time-series cross-validation
- Multiple regression models
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.sunspot_data import get_feature_matrix, get_feature_matrix_from_start_year, LAGS
from src.models.registry import get_model, AVAILABLE_MODELS
from src.results_manager import save_results_to_csv, save_predictions_to_csv


# Constants from requirements
TEST_WINDOW_MONTHS = 144  # 2004-2015: 12 years of test data
N_FOLDS = 6
START_YEAR_MIN = 1749
START_YEAR_MAX = 1940  # Training data up to starting at 1940


def split_train_test(
    X: pd.DataFrame, y: pd.Series, test_months: int = TEST_WINDOW_MONTHS
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split data into training and test sets.
    
    CRITICAL: This split happens AFTER lag features are created.
    The test set is the last `test_months` rows of the lagged dataset.
    
    The test set must NEVER appear in training for any configuration.
    
    Args:
        X: Feature matrix with DatetimeIndex (already has lag features)
        y: Target series with DatetimeIndex
        test_months: Number of months to reserve for testing (default: 131)
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    # Ensure chronological order
    X = X.sort_index()
    y = y.sort_index()
    
    # Test set: last 131 rows (AFTER lag features are built)
    X_test = X.iloc[-test_months:]
    y_test = y.iloc[-test_months:]
    
    # Training region: everything BEFORE the test set
    X_train = X.iloc[:-test_months]
    y_train = y.iloc[:-test_months]
    
    return X_train, y_train, X_test, y_test


def filter_by_start_year(
    X: pd.DataFrame, y: pd.Series, start_year: int
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Filter TRAINING data to only include records from start_year onwards.
    
    CRITICAL: This should only be called on the training region, NEVER on test set.
    The test set remains completely fixed across all start_year configurations.
    
    Args:
        X: Training feature matrix with DatetimeIndex
        y: Training target series with DatetimeIndex
        start_year: First year to include in training
    
    Returns:
        Filtered X_train, y_train
    """
    # Filter by year
    mask = X.index.year >= start_year  # type: ignore[attr-defined]
    return X[mask], y[mask]


def time_series_cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_splits: int = N_FOLDS,
) -> tuple[float, list[float]]:
    """
    Perform time-series cross-validation.
    
    Each fold preserves temporal ordering:
    - Train on first 80% of fold
    - Validate on remaining 20%
    
    Args:
        X: Training features
        y: Training target
        model_name: Name of model to use (from registry)
        n_splits: Number of CV folds
    
    Returns:
        mean_rmse: Average RMSE across folds
        fold_rmses: List of RMSE for each fold
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmses: list[float] = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Get fold data
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_val = y.iloc[val_idx]
        
        # Train model
        model = get_model(model_name)
        model.fit(X_fold_train, y_fold_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_fold_val)
        rmse = float(np.sqrt(mean_squared_error(y_fold_val, y_pred)))
        fold_rmses.append(rmse)
    
    mean_rmse = float(np.mean(fold_rmses))
    return mean_rmse, fold_rmses


def evaluate_model(
    model_name: str,
    start_year: int,
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True,
    skip_cv: bool = False,
) -> dict[str, Any]:
    """
    Evaluate a single (model, start_year) configuration.
    
    Steps:
    1. Filter training data by start_year
    2. Perform 6-fold time-series CV to get validation RMSE
    3. Retrain on full filtered training data
    4. Evaluate on test set
    
    Args:
        model_name: Name of model (from registry)
        start_year: First year to include in training
        X_train_full: Full training features
        y_train_full: Full training target
        X_test: Test features
        y_test: Test target
        verbose: Print progress
        skip_cv: Skip cross-validation if True
    
    Returns:
        Dictionary with results:
            - model_name
            - start_year
            - n_train_samples
            - cv_mean_rmse
            - cv_fold_rmses
            - test_rmse
            - model (trained on full filtered training data)
    """
    # Filter by start year
    X_train, y_train = filter_by_start_year(X_train_full, y_train_full, start_year)
    
    if verbose:
        print(f"  Start year {start_year}: {len(X_train)} training samples", end="")
    
    # Cross-validation (optional)
    if skip_cv:
        cv_mean_rmse = None
        cv_fold_rmses = None
    else:
        cv_mean_rmse, cv_fold_rmses = time_series_cross_validate(
            X_train, y_train, model_name, n_splits=N_FOLDS
        )
    
    if verbose and cv_mean_rmse is not None:
        print(f" | CV RMSE: {cv_mean_rmse:.2f}", end="")
    
    # Retrain on full training data (for this start year)
    model = get_model(model_name)
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    if verbose:
        print(f" | Test RMSE: {test_rmse:.2f}")
    
    return {
        "model_name": model_name,
        "start_year": start_year,
        "n_train_samples": len(X_train),
        "cv_mean_rmse": cv_mean_rmse,
        "cv_fold_rmses": cv_fold_rmses,
        "test_rmse": test_rmse,
        "model": model,
        "predictions": y_pred_test,
    }


def find_best_configuration(
    results: list[dict],
    metric: str = "cv_mean_rmse",
) -> dict:
    """
    Find the best (model, start_year) configuration.
    
    Args:
        results: List of result dictionaries from evaluate_model
        metric: Metric to minimize ("cv_mean_rmse" or "test_rmse")
    
    Returns:
        Best result dictionary
    """
    if not results:
        raise ValueError("No results to evaluate")
    
    best_result = min(results, key=lambda r: r[metric])
    return best_result


def run_experiment(
    model_names: list[str] | None = None,
    start_years: list[int] | None = None,
    verbose: bool = True,
    save_to_csv: bool = True,
    save_best_predictions: bool = True,
    skip_cv: bool = False,
    apply_merit_weights: bool = False,
) -> tuple[list[dict], dict]:
    """
    Run the full sunspot prediction experiment.
    
    Args:
        model_names: List of model names to test (default: all available)
        start_years: List of start years to test (default: 1749-1940, step 10)
        verbose: Print progress
        save_to_csv: Save results to CSV file
        save_best_predictions: Save predictions from best model to CSV
        skip_cv: Skip cross-validation and only use test RMSE
        apply_merit_weights: Apply Merit Attribute weighting to lag features
    
    Returns:
        results: List of all evaluation results
        best_result: Best (model, start_year) configuration
    """
    # Default parameters
    if model_names is None:
        model_names = AVAILABLE_MODELS
    
    if start_years is None:
        # Test every 10 years from 1749 to 1940
        start_years = list(range(START_YEAR_MIN, START_YEAR_MAX + 1, 10))
    
    # Load data
    if verbose:
        merit_status = "WITH Merit Attribute weighting" if apply_merit_weights else "WITHOUT Merit weighting"
        print(f"Loading data... ({merit_status})")
    X, y = get_feature_matrix(LAGS, apply_merit_weights=apply_merit_weights)
    
    # Split train/test
    X_train_full, y_train_full, X_test, y_test = split_train_test(X, y)
    
    if verbose:
        print(f"Dataset: {len(X)} total samples")
        print(f"Training region: {len(X_train_full)} samples")
        print(f"Test region: {len(X_test)} samples ({TEST_WINDOW_MONTHS} months)")
        print(f"Test period: {X_test.index[0].date()} to {X_test.index[-1].date()}")
        print()
    
    # Run experiments
    results = []
    
    for model_name in model_names:
        if verbose:
            print(f"Model: {model_name}")
        
        for start_year in start_years:
            result = evaluate_model(
                model_name=model_name,
                start_year=start_year,
                X_train_full=X_train_full,
                y_train_full=y_train_full,
                X_test=X_test,
                y_test=y_test,
                verbose=verbose,
                skip_cv=skip_cv,
            )
            results.append(result)
        
        if verbose:
            print()
    
    # Find best configuration (use test_rmse if CV was skipped)
    metric_to_use = "test_rmse" if skip_cv else "cv_mean_rmse"
    best_result = find_best_configuration(results, metric=metric_to_use)
    
    if verbose:
        print("=" * 70)
        if skip_cv:
            print("BEST CONFIGURATION (by Test RMSE):")
        else:
            print("BEST CONFIGURATION (by CV RMSE):")
        print(f"  Model: {best_result['model_name']}")
        print(f"  Start year: {best_result['start_year']}")
        print(f"  Training samples: {best_result['n_train_samples']}")
        if not skip_cv:
            print(f"  CV RMSE: {best_result['cv_mean_rmse']:.2f}")
        print(f"  Test RMSE: {best_result['test_rmse']:.2f}")
        print("=" * 70)
    
    # Save results to CSV
    if save_to_csv:
        csv_path = save_results_to_csv(results)
        if verbose:
            print(f"\nResults saved to: {csv_path}")
    
    # Save best model predictions
    if save_best_predictions:
        pred_path = save_predictions_to_csv(
            best_result,
            pd.DatetimeIndex(X_test.index),
            y_test
        )
        if verbose:
            print(f"Best predictions saved to: {pred_path}")
    
    return results, best_result


def main():
    """
    Run a quick experiment with a subset of models and start years.
    """
    # Quick test with fewer configurations
    print("Running sunspot prediction experiments...\n")
    
    results, best = run_experiment(
        model_names=["linear_regression", "random_forest"],
        start_years=[1749, 1800, 1850, 1900, 1940],
        verbose=True,
    )
    
    print(f"\nTotal configurations tested: {len(results)}")


if __name__ == "__main__":
    main()
