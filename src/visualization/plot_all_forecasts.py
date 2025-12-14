"""
Generate a superimposed forecast plot showing all models' predictions.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from src.data.sunspot_data import get_feature_matrix, LAGS
from src.models.registry import get_model, DEFAULT_MODELS

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# Path configuration
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "forecasts"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_all_forecasts(
    model_names: list[str] | None = None,
    years_past: int = 15,
    years_future: int = 15,
    save: bool = True,
    show: bool = False,
    output_filename: str | None = None,
) -> None:
    """
    Generate a superimposed plot of all model forecasts.
    
    Args:
        model_names: List of model names to include. Defaults to DEFAULT_MODELS.
        years_past: Number of years of historical data to display
        years_future: Number of years to forecast into the future
        save: Whether to save the figure
        show: Whether to display the figure
        output_filename: Custom output filename (optional)
    """
    if model_names is None:
        model_names = DEFAULT_MODELS
    
    # Load data
    X, y = get_feature_matrix(LAGS)
    
    # Determine time range for display
    max_date = X.index.max()
    past_cutoff = max_date - pd.DateOffset(years=years_past)
    
    # Filter historical data for display
    mask = X.index >= past_cutoff
    X_display = X[mask]
    y_display = y[mask]
    
    # Define distinct colors for each model
    colors = {
        'linear_regression': '#1f77b4',  # blue
        'random_forest': '#2ca02c',      # green
        'xgboost': '#ff7f0e',            # orange
        'kneighbors': '#d62728',         # red
        'svm': '#9467bd',                # purple
        'gaussian_process': '#8c564b',   # brown
        'ridge': '#e377c2',              # pink
        'lasso': '#7f7f7f',              # gray
        'elastic_net': '#bcbd22',        # yellow-green
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 9))
    
    # Plot historical actual data (up to present only)
    ax.plot(
        y_display.index,
        y_display,
        label="Actual SSN",
        linewidth=3,
        alpha=0.9,
        color="navy",
        marker="o",
        markersize=4,
        zorder=10,
    )
    
    # Generate forecasts for each model
    months_future = years_future * 12
    
    for model_name in model_names:
        print(f"Generating forecast for {model_name}...")
        
        # Train model on all data
        model = get_model(model_name)
        model.fit(X, y)
        
        # Generate future forecasts
        future_dates = []
        future_predictions = []
        
        # Start from the last known values
        last_values = y.iloc[-max(LAGS):].values
        last_date = X.index.max()
        
        # Generate monthly forecasts
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
            
            # Update last_values with prediction
            last_values = np.append(last_values, np.float64(next_pred))  # type: ignore[arg-type]
        
        # Plot this model's forecast
        model_label = model_name.replace('_', ' ').title()
        model_color = colors.get(model_name, f'C{len(ax.lines)}')
        
        ax.plot(
            future_dates,
            future_predictions,
            label=f"{model_label} Forecast",
            linewidth=2.5,
            alpha=0.85,
            color=model_color,
            marker='s',
            markersize=2,
            markevery=12,  # Mark every year
        )
    
    # Add vertical line at present
    ax.axvline(x=max_date, color="black", linestyle=":", linewidth=2.5, 
               alpha=0.8, label="Present", zorder=5)
    
    # Labels and styling
    ax.set_xlabel("Date", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sunspot Number", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Sunspot Forecasts: All Models Comparison\n"
        f"({years_past} Years Historical + {years_future} Years Forecast)",
        fontsize=16,
        fontweight="bold",
        pad=20
    )
    ax.legend(fontsize=11, loc="best", framealpha=0.95, shadow=True)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    
    # Add subtle styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    if save:
        if output_filename is None:
            # Use year ranges in filename
            hist_start_year = int(X_display.index[0].year)
            future_end_year = int(future_dates[-1].year) # type: ignore
            output_filename = f"all_forecasts_{hist_start_year}_{future_end_year}.png"
        output_path = FIGURES_DIR / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate forecast plots")
    parser.add_argument("--models", nargs="+", help="Specific models to plot (default: all)")
    parser.add_argument("--years-past", type=int, default=5, help="Years of history to show (default: 5)")
    parser.add_argument("--years-future", type=int, default=10, help="Years to forecast (default: 10)")
    parser.add_argument("--output", help="Custom output filename")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    
    args = parser.parse_args()
    
    plot_all_forecasts(
        model_names=args.models,
        years_past=args.years_past,
        years_future=args.years_future,
        save=True,
        show=args.show,
        output_filename=args.output,
    )
