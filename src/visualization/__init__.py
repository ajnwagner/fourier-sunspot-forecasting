# src/visualization/__init__.py

# Import functions lazily to avoid RuntimeWarning when running modules with python -m
# from .plot_rmse_vs_year import plot_rmse_vs_year
# from .plot_training_fit import plot_training_fit
# from .plot_forecast import plot_forecast
# from .plot_all_forecasts import plot_all_forecasts
# from .plot_forecast_grid import plot_forecast_grid
# from .plot_accuracy_summary import plot_accuracy_summary

__all__ = [
    # Individual plot functions
    "plot_rmse_vs_year",
    "plot_training_fit",
    "plot_forecast",
    "plot_all_forecasts",
    "plot_forecast_grid",
    "plot_accuracy_summary",
]
