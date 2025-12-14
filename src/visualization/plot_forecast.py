# src/visualization/plot_forecast.py

"""
Plot historical and forecasted sunspot numbers for a specific model.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "forecasts"


def plot_forecast(
    model_name: str,
    years_past: int = 10,
    years_future: int = 10,
    save: bool = True,
    show: bool = False,
    output_filename: str | None = None,
) -> None:
    """
    Plot historical data and future forecast for a specific model.
    
    Shows real data up to present and forecasted data extending into the future.
    
    Args:
        model_name: Name of model to use for forecasting
        years_past: Years of historical data to show
        years_future: Years to forecast into future
        save: Save figure to disk
        show: Display figure interactively
        output_filename: Custom output filename
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    
    from src.data.sunspot_data import get_feature_matrix, LAGS
    from src.models.registry import get_model
    
    FIGURES_DIR.mkdir(exist_ok=True)
    
    # Load all available data
    X, y = get_feature_matrix(LAGS)
    
    # Train on all data
    model = get_model(model_name)
    model.fit(X, y)
    
    # Get predictions on all historical data
    y_pred_historical = model.predict(X)
    
    # Determine time range for display
    max_date = X.index.max()
    past_cutoff = max_date - pd.DateOffset(years=years_past)
    
    # Filter historical data for display
    mask = X.index >= past_cutoff
    X_display = X[mask]
    y_display = y[mask]
    y_pred_display = y_pred_historical[mask]
    
    # Generate future forecasts
    # We'll use an iterative approach, feeding predictions back as features
    future_dates = []
    future_predictions = []
    
    # Start from the last known values
    last_values = y.iloc[-max(LAGS):].values  # Get enough history for all lags
    last_date = X.index.max()
    
    # Generate monthly forecasts
    months_future = years_future * 12
    for i in range(months_future):
        # Create feature vector from last values
        features = np.array([[
            last_values[-lag] if lag <= len(last_values) else last_values[0]
            for lag in LAGS
        ]])
        
        # Predict next value
        next_pred = model.predict(features)[0]
        
        # Store prediction
        next_date = last_date + pd.DateOffset(months=i+1)
        future_dates.append(next_date)
        future_predictions.append(next_pred)
        
        # Update last_values with prediction (cast to fix typing)
        last_values = np.append(last_values, np.float64(next_pred))  # type: ignore[arg-type]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot historical actual data
    ax.plot(
        y_display.index,
        y_display,
        label="Actual SSN (Historical)",
        linewidth=2.5,
        alpha=0.9,
        color="navy",
        marker="o",
        markersize=3,
    )
    
    # Plot historical model fit (only up to present)
    ax.plot(
        X_display.index,
        y_pred_display,
        label="Model Fit (Historical)",
        linewidth=2,
        alpha=0.7,
        linestyle="--",
        color="green",
    )
    
    # Plot future forecast
    ax.plot(
        future_dates,
        future_predictions,
        label=f"Forecast ({years_future} Years)",
        linewidth=2.5,
        alpha=0.8,
        linestyle="-",
        color="crimson",
        marker="s",
        markersize=3,
    )
    
    # Add vertical line at present
    ax.axvline(x=max_date, color="black", linestyle=":", linewidth=2, 
               alpha=0.7, label="Present")
    
    # Shade future region
    ax.axvspan(max_date, future_dates[-1], alpha=0.1, color="red", 
               label="Forecast Region")
    
    # Labels and styling
    ax.set_xlabel("Date", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sunspot Number", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Sunspot Forecast: {model_name.replace('_', ' ').title()}\n"
        f"({years_past} Years Historical + {years_future} Years Forecast)",
        fontsize=14,
        fontweight="bold",
        pad=15
    )
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    # Add confidence interval shading for forecast (simple ±1 std approximation)
    forecast_std = np.std(y_pred_display - y_display.values)
    ax.fill_between(
        future_dates,
        np.array(future_predictions) - forecast_std,
        np.array(future_predictions) + forecast_std,
        alpha=0.2,
        color="crimson",
    )
    
    plt.tight_layout()
    
    if save:
        if output_filename is None:
            # Use exact year ranges in filename (historical start to future end)
            hist_start_year = int(X_display.index[0].year)  # type: ignore
            future_end_year = int(future_dates[-1].year)
            output_filename = f"forecast_{model_name}_{hist_start_year}_{future_end_year}.png"
        output_path = FIGURES_DIR / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Example usage
    plot_forecast("random_forest", years_past=10, years_future=10, save=True, show=False)
