#!/usr/bin/env python3
"""
Generate all publication plots without runtime warnings.
"""

import warnings
warnings.filterwarnings('ignore')

from src.visualization.plot_rmse_vs_year import plot_rmse_vs_year
from src.visualization.plot_accuracy_summary import plot_accuracy_summary
from src.visualization.plot_forecast import plot_forecast
from src.visualization.plot_all_forecasts import plot_all_forecasts
from src.results_manager import get_latest_results

def main():
    print("Generating publication figures...")
    print("=" * 60)
    
    # Get latest results
    latest = get_latest_results()
    if not latest:
        print("Error: No results found. Run experiments first.")
        return
    
    print(f"\nUsing results: {latest.name}")
    print()
    
    # 1. RMSE vs Year plot
    print("1. Generating RMSE vs Training Start Year plot...")
    plot_rmse_vs_year(latest, metric="test_rmse", save=True, show=False)
    
    # 2. Accuracy summary table
    print("2. Generating accuracy summary table...")
    plot_accuracy_summary(latest, save=True, show=False)
    
    # 3. Individual model forecasts
    models = ['linear_regression', 'random_forest', 'xgboost', 'kneighbors', 'svm']
    for i, model in enumerate(models, 3):
        print(f"{i}. Generating {model.replace('_', ' ').title()} forecast...")
        plot_forecast(model, years_past=10, years_future=10, save=True, show=False)
    
    # 4. Combined overlay
    print("8. Generating combined forecast overlay...")
    plot_all_forecasts(years_past=5, years_future=10, save=True, show=False)
    
    print()
    print("=" * 60)
    print("All publication figures generated successfully!")
    print("Check figures/rmse_analysis/ and figures/forecasts/")

if __name__ == "__main__":
    main()
