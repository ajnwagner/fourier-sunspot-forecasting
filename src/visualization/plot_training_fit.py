# src/visualization/plot_training_fit.py

"""
Plot actual vs. predicted sunspot numbers during training for a specific model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "training_fits"


def plot_training_fit(
    model_name: str,
    last_n_years: int,
    save: bool = True,
    show: bool = False,
    output_filename: str | None = None,
) -> None:
    """
    Plot actual vs. predicted sunspot numbers for training period.
    
    This requires re-running the model to get training predictions.
    
    Args:
        model_name: Name of model to evaluate
        last_n_years: How many years of training data to use
        save: Save figure to disk
        show: Display figure interactively
        output_filename: Custom output filename
    """
    from src.data.sunspot_data import get_feature_matrix, LAGS
    from src.models.registry import get_model
    
    FIGURES_DIR.mkdir(exist_ok=True)
    
    # Load data
    X, y = get_feature_matrix(LAGS)
    
    # Calculate cutoff for last N years
    max_date = X.index.max()
    cutoff_date = max_date - pd.DateOffset(years=last_n_years)
    
    # Filter to last N years
    mask = X.index >= cutoff_date
    X_train = X[mask]
    y_train = y[mask]
    
    # Train model
    model = get_model(model_name)
    model.fit(X_train, y_train)
    
    # Get predictions on training data
    y_pred = model.predict(X_train)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Time series comparison
    ax1.plot(X_train.index, y_train, label="Actual SSN", linewidth=2, alpha=0.8, color="navy")
    ax1.plot(X_train.index, y_pred, label="Predicted SSN", linewidth=2, alpha=0.8, 
             linestyle="--", color="crimson")
    ax1.fill_between(
        X_train.index,
        y_train,
        y_pred,
        alpha=0.2,
        color="orange",
        label="Error Region"
    )
    
    ax1.set_xlabel("Date", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Sunspot Number", fontsize=13, fontweight="bold")
    ax1.set_title(
        f"Training Fit: {model_name.replace('_', ' ').title()} (Last {last_n_years} Years)",
        fontsize=14,
        fontweight="bold",
        pad=15
    )
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(True, alpha=0.3, linestyle="--")
    
    # Plot 2: Scatter plot (actual vs predicted)
    ax2.scatter(y_train, y_pred, alpha=0.5, s=30, color="steelblue")
    
    # Perfect prediction line (cast to fix typing)
    min_val = min(float(y_train.min()), float(y_pred.min()))  # type: ignore[union-attr]
    max_val = max(float(y_train.max()), float(y_pred.max()))  # type: ignore[union-attr]
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label="Perfect Prediction", alpha=0.7)
    
    # Calculate R²
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    
    ax2.set_xlabel("Actual SSN", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Predicted SSN", fontsize=13, fontweight="bold")
    ax2.set_title(f"Prediction Accuracy (R² = {r2:.3f}, RMSE = {rmse:.2f})", 
                  fontsize=14, fontweight="bold", pad=15)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    
    if save:
        if output_filename is None:
            # Use exact year range in filename
            min_year = int(X_train.index.year.min())
            max_year = int(X_train.index.year.max())
            output_filename = f"training_fit_{model_name}_{min_year}_{max_year}.png"
        output_path = FIGURES_DIR / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Example usage
    plot_training_fit("random_forest", last_n_years=50, save=True, show=False)
