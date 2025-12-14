# src/data/sunspot_data.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Change this if your filename is different
DEFAULT_CSV = DATA_DIR / "sunspots.csv"

# Fourier-based lag specification from the paper
# These correspond to dominant solar-cycle periods:
# 5.45, 8.52, 10.91, 15.15, 30.4, 54.53 years → 65, 102, 131, 182, 365, 654 months
# Long lags are essential for capturing solar cycle memory
# The 654-month lag naturally limits early start years - this is correct behavior
LAGS = [65, 102, 131, 182, 365, 654]

# Merit Attribute weights from the paper (Table in original study)
# These represent the data-driven importance ranking of each lag
# Used to weight the contribution of different solar cycle harmonics
MERIT_WEIGHTS = {
    65: 8.045,
    102: 2.550,
    131: 14.696,
    182: 4.053,
    365: 1.0,  # Not specified in paper, using neutral weight
    654: 0.261,
}


def load_sunspot_csv(csv_path: Path = DEFAULT_CSV) -> pd.DataFrame:
    """
    Load the SILSO sunspot CSV and return a DataFrame with a DatetimeIndex
    and a single 'ssn' column containing the (smoothed) sunspot number.

    This tries to be robust to different CSV column names:
    - If there's a 'date' column, it parses that.
    - Else, if there are 'year' and 'month' columns, it builds a date.
    - Else, it assumes the first two columns are year and month.

    For the sunspot value, it prefers known names like:
    ['smoothed', 'ssn_smoothed', 'ssn', 'sunspot_number', 'monthly_smoothed']
    and falls back to the third column if needed.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path, sep=";")

    # --- Build date column ---
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif {"year", "month"}.issubset(df.columns):
        df["date"] = pd.to_datetime(
            pd.DataFrame({"year": df["year"], "month": df["month"], "day": 15})
        )
    else:
        # Assume first two columns are year and month
        year_col = df.columns[0]
        month_col = df.columns[1]
        df["date"] = pd.to_datetime(
            pd.DataFrame({"year": df[year_col], "month": df[month_col], "day": 15})
        )

    # --- Determine which column holds the smoothed SSN ---
    value_col_candidates = [
        "smoothed",
        "smoothed_ssn",
        "ssn_smoothed",
        "ssn",
        "sunspot_number",
        "monthly_smoothed",
    ]

    value_col = None
    for c in value_col_candidates:
        if c in df.columns:
            value_col = c
            break

    # Fallback: assume the 3rd column is the value
    if value_col is None:
        if len(df.columns) < 3:
            raise ValueError(
                "Could not infer value column. "
                "Make sure the CSV has at least 3 columns, or rename the "
                "sunspot column to something like 'smoothed' or 'ssn'."
            )
        value_col = df.columns[2]

    # --- Clean and return series ---
    series = (
        df[["date", value_col]]
        .rename(columns={value_col: "ssn"})
        .sort_values("date")
        .set_index("date")
    )
    
    # Filter out invalid values (as per requirements)
    series = series[series["ssn"] > 0].copy()

    return series


def load_sunspot_data(csv_path: Path = DEFAULT_CSV) -> pd.DataFrame:
    """
    Load sunspot data according to project specifications.
    Alias for load_sunspot_csv with filtering of invalid values.
    
    Returns DataFrame with DatetimeIndex and 'ssn' column,
    with all ssn <= 0 values removed.
    """
    return load_sunspot_csv(csv_path)


def add_lag_features(
    ssn_df: pd.DataFrame,
    lags: Iterable[int] = (65, 102, 131, 182, 654),
    apply_merit_weights: bool = False,
) -> pd.DataFrame:
    """
    Given a DataFrame with index=date and column 'ssn',
    add lagged versions of the series as additional columns.

    Lags are in *months* (i.e., record positions), matching the paper:
    5.45, 8.52, 10.91, 15.15, 54.53 years ≈ 65, 102, 131, 182, 654 months.
    
    Args:
        ssn_df: DataFrame with DatetimeIndex and 'ssn' column
        lags: Lag values in months
        apply_merit_weights: If True, multiply each lag feature by its Merit Attribute weight
    """
    if "ssn" not in ssn_df.columns:
        raise ValueError("Input DataFrame must have an 'ssn' column.")

    df = ssn_df.copy()
    for lag in lags:
        df[f"ssn_lag_{lag}"] = df["ssn"].shift(lag)
        
        # Apply Merit Attribute weighting if requested
        if apply_merit_weights and lag in MERIT_WEIGHTS:
            df[f"ssn_lag_{lag}"] *= MERIT_WEIGHTS[lag]

    # Drop rows where any lag is NaN (front of the series)
    df = df.dropna()

    return df


def get_feature_matrix(
    lags: Iterable[int] = LAGS,
    apply_merit_weights: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load clean sunspot data and generate feature matrix.
    
    Args:
        lags: Lag values in months
        apply_merit_weights: If True, apply Merit Attribute weighting to lag features
    
    Returns:
        X: DataFrame with lagged features only (columns: ssn_lag_65, ssn_lag_102, etc.)
        y: Series with current ssn values
    """
    # Load clean data
    ssn_df = load_sunspot_data()
    
    # Add lag features (with optional Merit weighting)
    lagged_df = add_lag_features(ssn_df, lags, apply_merit_weights)
    
    # Split into features (X) and target (y)
    lag_columns = [f"ssn_lag_{lag}" for lag in lags]
    X = lagged_df[lag_columns]
    y = lagged_df["ssn"]
    
    return X, y


def get_feature_matrix_from_start_year(
    start_year: int, 
    lags: Iterable[int] = LAGS,
    apply_merit_weights: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load sunspot data starting from a specific year and generate feature matrix.
    
    This ensures that different start years produce different training sets.
    The key difference from get_feature_matrix() is that we filter by start_year
    BEFORE adding lag features, not after.
    
    Args:
        start_year: First year to include in the dataset
        lags: Lag values in months
        apply_merit_weights: If True, apply Merit Attribute weighting to lag features
    
    Returns:
        X: DataFrame with lagged features
        y: Series with current ssn values
    """
    # Load clean data
    ssn_df = load_sunspot_data()
    
    # Filter by start year FIRST, before creating lags
    mask = ssn_df.index.year >= start_year  # type: ignore[attr-defined]
    ssn_filtered = ssn_df[mask].copy()
    
    # Now add lag features to the filtered data (with optional Merit weighting)
    lagged_df = add_lag_features(ssn_filtered, lags, apply_merit_weights)
    
    # Split into features (X) and target (y)
    lag_columns = [f"ssn_lag_{lag}" for lag in lags]
    X = lagged_df[lag_columns]
    y = lagged_df["ssn"]
    
    return X, y


def main() -> None:
    """
    Quick sanity check: load the data, build lag features,
    and print basic info. This is just to confirm wiring.
    """
    ssn = load_sunspot_data()
    print("Raw series (after filtering ssn > 0):")
    print(ssn.head(10))
    print()

    lagged = add_lag_features(ssn)
    print("Lagged feature frame:")
    print(lagged.head())
    print()

    print(f"Total monthly records: {len(ssn)}")
    print(f"Usable rows after lagging: {len(lagged)}")
    print()
    
    # Test get_feature_matrix
    X, y = get_feature_matrix()
    print(f"Feature matrix X shape: {X.shape}")
    print(f"Target y shape: {y.shape}")
    print("\nFirst few X rows:")
    print(X.head())


if __name__ == "__main__":
    main()
