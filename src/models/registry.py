# src/models/registry.py

"""
Model registry for sunspot prediction experiments.

Provides a centralized interface for creating sklearn-compatible regression models.
Allows easy experimentation with different model types without changing the
experiment code.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_model(name: str):
    """
    Get a regression model by name.
    
    Args:
        name: Model identifier. Supported values:
            - "linear_regression": Ridge regression (alpha=1.0)
            - "ridge": Ridge regression (alpha=10.0)
            - "lasso": Lasso regression
            - "elastic_net": ElasticNet regression
            - "random_forest": RandomForestRegressor with 100 trees
            - "xgboost": XGBoost regressor
            - "svm": Support Vector Regression with RBF kernel
            - "gaussian_process": Gaussian Process Regressor with RBF kernel
    
    Returns:
        Sklearn/XGBoost estimator with .fit() and .predict() methods
    
    Raises:
        ValueError: If model name is not recognized
    """
    name = name.lower().strip()
    
    if name == "linear_regression":
        # Using Ridge with alpha=1.0 for slight regularization
        return Ridge(alpha=1.0, random_state=42)
    
    elif name == "ridge":
        return Ridge(alpha=10.0, random_state=42)
    
    elif name == "lasso":
        return Lasso(alpha=1.0, random_state=42, max_iter=10000)
    
    elif name == "elastic_net":
        return ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)
    
    elif name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    
    elif name == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        return XGBRegressor(  # type: ignore[possibly-unbound]
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        )
    
    elif name == "kneighbors":
        return KNeighborsRegressor(
            n_neighbors=5,
            weights="uniform",
            n_jobs=-1,
        )
    
    elif name == "svm":
        return SVR(
            kernel="rbf",
            C=100.0,
            gamma="scale",
            epsilon=0.1
        )
    
    elif name == "gaussian_process":
        kernel = C(100.0, (1e-1, 1e5)) * RBF(100.0, (1e-1, 1e4))
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1.0,
            random_state=42,
            n_restarts_optimizer=5,
            normalize_y=True
        )
    
    else:
        raise ValueError(
            f"Unknown model '{name}'. Available models: {AVAILABLE_MODELS}"
        )


# All available model names
AVAILABLE_MODELS = [
    "elastic_net",
    "gaussian_process",
    "kneighbors",
    "lasso",
    "linear_regression",
    "random_forest",
    "ridge",
    "svm",
    "xgboost",
]

# Default models (core set for typical experiments)
DEFAULT_MODELS = [
    "kneighbors",
    "linear_regression",
    "random_forest",
    "svm",
    "xgboost",
]
