# Sunspot Prediction Experiment

> **Note:** This README was generated with AI assistance and may require updates for your specific setup.

A modular, reproducible Python pipeline for sunspot forecasting using SILSO V2.0 monthly sunspot data. This project trains multiple machine learning models with Fourier-based lag features and cross-validation to predict solar cycle activity.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Makefile Commands](#makefile-commands)
5. [Python Scripts & Flags](#python-scripts--flags)
6. [Methodology](#methodology)
7. [Customization](#customization)
8. [File Outputs](#file-outputs)

---

## Quick Start

**Complete workflow in 3 commands:**

```bash
make install   # Install dependencies
make train     # Train all models (1804-1940, CV, merit weights)
make publish   # Generate all publication figures
```

All publication-ready figures will be in `figures/publish/`.

---

## Project Structure

```
aml-final-project/
├── data/
│   └── sunspots.csv                      # SILSO V2.0 monthly sunspot data
│
├── results/                              # Experiment outputs
│   ├── experiment_results.csv            # Primary results (git-tracked)
│   ├── experiment_results_*.csv          # Timestamped results
│   └── *_predictions_*.csv               # Model predictions
│
├── figures/                              # Generated visualizations
│   ├── publish/                          # Publication-ready (git-tracked)
│   ├── rmse_analysis/                    # RMSE vs year plots
│   ├── training_fits/                    # Training fit plots
│   └── forecasts/                        # Forecast plots
│
├── src/
│   ├── data/
│   │   └── sunspot_data.py               # Data loading & feature engineering
│   ├── models/
│   │   └── registry.py                   # Model definitions
│   ├── experiments/
│   │   └── run_sunspot_experiments.py    # Main experiment runner
│   ├── visualization/
│   │   ├── plot_rmse_vs_year.py          # RMSE analysis
│   │   ├── plot_training_fit.py          # Training fits
│   │   ├── plot_forecast.py              # Individual forecasts
│   │   ├── plot_all_forecasts.py         # Overlay forecasts
│   │   ├── plot_forecast_grid.py         # Grid layouts
│   │   └── plot_accuracy_summary.py      # Summary tables
│   └── results_manager.py                # Result I/O utilities
│
├── main.py                               # Command-line interface
├── requirements.txt                      # Python dependencies
├── Makefile                              # Build automation
└── README.md                             # This file
```

---

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
make install
# or
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
make verify  # Tests data loading
```

**Requirements:**
- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- xgboost >= 2.0.0

---

## Makefile Commands

### Main Workflow

| Command | Description | Usage |
|---------|-------------|-------|
| `make install` | Install all dependencies | Run once after cloning |
| `make train` | Train all models (1804-1940, CV, merit weights) | ~30-60 min runtime |
| `make publish` | Generate all publication figures | Run after training |

**Example workflow:**
```bash
make install   # First time setup
make train     # Train models (creates results/experiment_results.csv)
make publish   # Generate figures → figures/publish/
```

### Modular Visualization

Generate specific figure types without retraining:

| Command | Description | Output Location |
|---------|-------------|-----------------|
| `make forecast` | All model forecast plots (2020-2035) | `figures/forecasts/forecast_*_2020_2035.png` |
| `make training-fit` | All model training fits (30 years) | `figures/training_fits/training_fit_*_1995_2025.png` |
| `make rmse` | Test & CV RMSE vs year (1804-1940) | `figures/rmse_analysis/rmse_vs_year_*.png` |

**Example:**
```bash
make rmse      # Just regenerate RMSE plots
make forecast  # Just regenerate forecast plots
```

### Utility Commands

| Command | Description |
|---------|-------------|
| `make verify` | Verify data loading works |
| `make quick` | Quick test (2 models, few years, no CV) |
| `make clean` | Remove `__pycache__` and temp files |
| `make zip` | Create `../aml-final-project.zip` from git files |
| `make help` | Show all available commands |

---

## Python Scripts & Flags

All scripts can be run directly with Python for fine-grained control.

### 1. Training Experiments (`main.py`)

**Basic usage:**
```bash
python main.py --dense 1804 1940 --cv --merit-weights
```

**All flags:**

| Flag | Description | Example |
|------|-------------|---------|
| `--dense START END` | Train on every year from START to END | `--dense 1804 1940` |
| `--start-years Y1 Y2 ...` | Specific start years | `--start-years 1800 1850 1900` |
| `--models M1 M2 ...` | Specific models to train | `--models xgboost linear_regression` |
| `--cv` | Enable 6-fold cross-validation | |
| `--merit-weights` | Apply Merit Attribute weights to features | |
| `--mark-primary` | Copy results to `experiment_results.csv` | |
| `--quiet` | Reduce console output | |

**Examples:**
```bash
# Full experiment (same as make train)
python main.py --dense 1804 1940 --cv --merit-weights --mark-primary

# Train only XGBoost on specific years
python main.py --start-years 1850 1900 1950 --models xgboost --cv

# Quick test without CV
python main.py --start-years 1900 --models linear_regression
```

**Default models:** `linear_regression`, `random_forest`, `xgboost`, `kneighbors`, `svm`

---

### 2. RMSE vs Year Plots (`plot_rmse_vs_year.py`)

**Basic usage:**
```bash
python -m src.visualization.plot_rmse_vs_year results/experiment_results.csv --metric cv --start-year 1804 --end-year 1940
```

**All flags:**

| Flag | Description | Example |
|------|-------------|---------|
| `csv_path` | Path to results CSV (optional) | `results/experiment_results.csv` |
| `--metric {test,cv}` | Plot test RMSE or CV RMSE | `--metric cv` |
| `--start-year YEAR` | Filter to start year >= YEAR | `--start-year 1804` |
| `--end-year YEAR` | Filter to start year <= YEAR | `--end-year 1940` |
| `--output FILE` | Custom output filename | `--output my_rmse.png` |
| `--show` | Display plot interactively | |

**Examples:**
```bash
# CV RMSE for 1804-1940
python -m src.visualization.plot_rmse_vs_year --metric cv --start-year 1804 --end-year 1940

# Test RMSE for full range (uses latest results)
python -m src.visualization.plot_rmse_vs_year --metric test

# Show plot instead of saving
python -m src.visualization.plot_rmse_vs_year --metric test --show
```

**Output:** `figures/rmse_analysis/rmse_vs_year_{test|cv}_YYYY_YYYY.png`

---

### 3. Training Fit Plots (`plot_training_fit.py`)

**Basic usage:**
```bash
python -c "from src.visualization.plot_training_fit import plot_training_fit; plot_training_fit('xgboost', last_n_years=30)"
```

**Parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `model_name` | Model to plot | `'xgboost'`, `'linear_regression'` |
| `last_n_years` | Training window size (years) | `30`, `50`, `100` |
| `save` | Save to file (default True) | `save=True` |
| `show` | Display interactively | `show=True` |

**Examples:**
```bash
# XGBoost last 30 years
python -c "from src.visualization.plot_training_fit import plot_training_fit; plot_training_fit('xgboost', last_n_years=30)"

# Linear regression last 50 years, show plot
python -c "from src.visualization.plot_training_fit import plot_training_fit; plot_training_fit('linear_regression', last_n_years=50, show=True)"
```

**Output:** `figures/training_fits/training_fit_MODEL_YYYY_YYYY.png`

---

### 4. Individual Forecast Plots (`plot_forecast.py`)

**Basic usage:**
```bash
python -m src.visualization.plot_forecast xgboost 2020 2035
```

**Arguments:**

| Argument | Description | Example |
|----------|-------------|---------|
| `model_name` | Model to forecast | `xgboost`, `linear_regression` |
| `start_year` | Forecast start year | `2020` |
| `end_year` | Forecast end year | `2035` |

**Additional flags:**

| Flag | Description |
|------|-------------|
| `--show` | Display plot interactively |

**Examples:**
```bash
# XGBoost forecast 2020-2035
python -m src.visualization.plot_forecast xgboost 2020 2035

# Linear regression forecast 2025-2040, display
python -m src.visualization.plot_forecast linear_regression 2025 2040 --show
```

**Output:** `figures/forecasts/forecast_MODEL_YYYY_YYYY.png`

---

### 5. Overlay All Forecasts (`plot_all_forecasts.py`)

**Basic usage:**
```bash
python -m src.visualization.plot_all_forecasts --models xgboost linear_regression --years-past 5 --years-future 10
```

**All flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--models M1 M2 ...` | Models to include | All 5 models |
| `--years-past N` | Years of history to show | 5 |
| `--years-future N` | Years to forecast | 10 |
| `--output FILE` | Custom filename | Auto-generated |
| `--show` | Display interactively | |

**Examples:**
```bash
# All models, 5 years past, 10 years future
python -m src.visualization.plot_all_forecasts

# Only XGBoost and Linear Regression, custom window
python -m src.visualization.plot_all_forecasts --models xgboost linear_regression --years-past 10 --years-future 15

# Just KNN forecast
python -m src.visualization.plot_all_forecasts --models kneighbors --years-future 20
```

**Output:** `figures/forecasts/all_forecasts_YYYY_YYYY.png`

---

### 6. Forecast Grid Layouts (`plot_forecast_grid.py`)

**Basic usage:**
```bash
python -m src.visualization.plot_forecast_grid --models linear_regression kneighbors xgboost
```

**All flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--models M1 M2 ...` | Models to include | `linear_regression kneighbors xgboost` |
| `--rows N` | Number of rows | Auto (= # models) |
| `--cols N` | Number of columns | 1 |
| `--output FILE` | Custom filename | `forecasts_grid_2020_2035.png` |
| `--show` | Display interactively | |

**Examples:**
```bash
# Default 3x1 grid (Linear, KNN, XGBoost)
python -m src.visualization.plot_forecast_grid

# All 5 models in 5x1 grid
python -m src.visualization.plot_forecast_grid --models linear_regression random_forest xgboost kneighbors svm

# 2x2 grid layout
python -m src.visualization.plot_forecast_grid --models xgboost linear_regression random_forest kneighbors --rows 2 --cols 2
```

**Output:** `figures/publish/forecasts_grid_2020_2035.png`

---

### 7. Accuracy Summary Table (`plot_accuracy_summary.py`)

**Basic usage:**
```bash
python -m src.visualization.plot_accuracy_summary
```

Automatically uses `results/experiment_results.csv` to generate a table showing:
- Model names (alphabetically sorted)
- Best training start year
- Test RMSE
- Test correlation

Best scores highlighted in **bold green**.

**Output:** `figures/rmse_analysis/accuracy_summary_table.png`

---

## Methodology

### Feature Engineering

**Fourier-Based Lag Features:**

Five lag features based on dominant solar cycle periods:
- **65 months** (5.45 years)
- **102 months** (8.52 years)
- **131 months** (10.91 years)
- **182 months** (15.15 years)
- **654 months** (54.53 years)

At time *t*: `X(t) = [ssn(t-65), ssn(t-102), ssn(t-131), ssn(t-182), ssn(t-654)]`

**Merit Attribute Weighting:**

When `--merit-weights` is enabled, each lag is multiplied by its importance weight:
- lag_131: **14.696** (most important)
- lag_65: **8.045**
- lag_182: **4.053**
- lag_102: **2.550**
- lag_365: **1.000**
- lag_654: **0.261**

### Train/Validation/Test Split

- **Test Set:** Fixed last 144 months (January 2004 - December 2015)
- **Training Window:** Filtered by start year (e.g., 1804-2003 for start_year=1804)
- **Cross-Validation:** 6-fold time-series split (80% train, 20% val per fold)

### Evaluation Metrics

For each (model, start_year) configuration:
1. **CV RMSE:** Mean RMSE across 6 validation folds
2. **Test RMSE:** RMSE on fixed 144-month test window
3. **Correlation:** Pearson correlation with actual SSN on test set

### Models

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| `linear_regression` | Ridge regression | α=1.0 |
| `random_forest` | Random Forest | 100 trees, max_depth=10 |
| `xgboost` | XGBoost | 100 estimators, depth=5 |
| `kneighbors` | K-Nearest Neighbors | k=5 |
| `svm` | Support Vector Regression | RBF kernel, C=1.0 |

---

## Customization

### Adding a New Model

1. **Edit** `src/models/registry.py`:

```python
def get_model(name: str):
    # ... existing models ...
    
    elif name == "my_custom_model":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            random_state=42
        )
```

2. **Update** `DEFAULT_MODELS` list:

```python
DEFAULT_MODELS = [
    "linear_regression",
    "random_forest",
    "xgboost",
    "kneighbors",
    "svm",
    "my_custom_model",  # Add here
]
```

3. **Run:**

```bash
python main.py --models my_custom_model --dense 1804 1940 --cv
```

### Modifying Lag Features

**Edit** `src/data/sunspot_data.py`:

```python
# Current lags (in months)
LAGS = [65, 102, 131, 182, 365, 654]

# Custom example: Add 24-month (2-year) lag
LAGS = [24, 65, 102, 131, 182, 365, 654]
```

### Changing Test Window

**Edit** `src/experiments/run_sunspot_experiments.py`:

```python
# Current: 144 months (2004-2015)
TEST_WINDOW_MONTHS = 144

# Example: 120 months (10 years)
TEST_WINDOW_MONTHS = 120
```

---

## File Outputs

### Results CSVs

**`results/experiment_results.csv`** (primary, git-tracked)
```
model_name,start_year,n_train_samples,cv_mean_rmse,test_rmse,cv_fold_1_rmse,...
linear_regression,1804,2503,47.23,18.68,45.12,...
xgboost,1878,2015,46.85,13.52,44.76,...
...
```

**Timestamped results:** `results/experiment_results_YYYYMMDD_HHMMSS.csv`

**Predictions:** `results/MODEL_yYEAR_predictions_TIMESTAMP.csv`
```
date,actual_ssn,predicted_ssn,error,absolute_error
2004-01-15,45.2,43.8,1.4,1.4
...
```

### Figure Outputs

**Publication folder:** `figures/publish/` (git-tracked)
- `rmse_vs_year_test_1804_1940.png`
- `rmse_vs_year_cv_1804_1940.png`
- `accuracy_summary_table.png`
- `forecast_MODEL_2020_2035.png` (5 files)
- `training_fit_MODEL_1995_2025.png` (5 files)
- `forecasts_grid_2020_2035.png`

**Analysis folders:** (gitignored, regenerated by `make publish`)
- `figures/rmse_analysis/` - RMSE plots
- `figures/training_fits/` - Training fit plots
- `figures/forecasts/` - Forecast plots

---

## License

This project is provided as-is for research and educational purposes.
python visualize_results.py
# or
make visualize
```
This reads results CSVs and generates three types of plots:
1. **RMSE Analysis** (`figures/rmse_analysis/`) - Performance vs. training start year
2. **Training Fits** (`figures/training_fits/`) - Model fit on training data
3. **Forecasts** (`figures/forecasts/`) - Historical + future predictions

### Generated Outputs

Experiments automatically save results to CSV files in `results/`:
- **experiment_results_*.csv** - All model configurations with CV and test RMSE
- ***_predictions_*.csv** - Predictions from the best model

Generated visualizations (saved to `figures/` subdirectories):
- **rmse_vs_year_*.png** - CV and Test RMSE vs. training start year
- **training_fit_*.png** - Actual vs. predicted during training
- **forecast_*.png** - Historical data + future forecasts

### Visualization Options

```bash
# Use most recent results
python visualize_results.py

# Use specific results file
python visualize_results.py results/experiment_results_20241214_120000.csv

# Display plots interactively
python visualize_results.py --show
```

## Usage

### Running Experiments

The main script supports various configurations:

```bash
# Default: quick test with linear_regression and random_forest
python run_experiments.py

# Test specific models
python run_experiments.py --models linear_regression svm

# Test all available models
python run_experiments.py --all-models

# Custom start years
python run_experiments.py --start-years 1800 1850 1900

# Full sweep (all models, all years)
python run_experiments.py --full

# Quiet mode (less output)
python run_experiments.py --quiet
```

### Available Models

- `linear_regression` - Ridge regression (α=1.0)
- `random_forest` - Random Forest with 100 trees
- `svm` - Support Vector Regression with RBF kernel
- `gaussian_process` - Gaussian Process with RBF kernel

### Data Module

```python
from src.data.sunspot_data import load_sunspot_data, get_feature_matrix, LAGS

# Load clean sunspot data
df = load_sunspot_data()

# Get feature matrix (X) and target (y)
X, y = get_feature_matrix()

# X contains 5 lagged features: ssn_lag_65, ssn_lag_102, ssn_lag_131, ssn_lag_182, ssn_lag_654
# y contains current SSN values
```

### Model Registry

```python
from src.models.registry import get_model

# Get a model by name
model = get_model("random_forest")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Custom Experiments

```python
from src.experiments.run_sunspot_experiments import run_experiment

# Run custom experiment
results, best = run_experiment(
    model_names=["linear_regression", "svm"],
    start_years=[1800, 1850, 1900],
    verbose=True,
    save_to_csv=True,  # Save results to CSV
    save_best_predictions=True,  # Save best model predictions
)

# Access results
print(f"Best model: {best['model_name']}")
print(f"Best start year: {best['start_year']}")
print(f"CV RMSE: {best['cv_mean_rmse']:.2f}")
print(f"Test RMSE: {best['test_rmse']:.2f}")
```

### Working with Results

```python
from src.results_manager import load_results_from_csv, get_latest_results
from src.visualization import plot_all_results

# Load results from CSV
results_df = load_results_from_csv("results/experiment_results_*.csv")

# Get most recent results
latest_csv = get_latest_results()

# Generate all plots
plot_all_results(latest_csv, save=True, show=False)
```

## Methodology

### Feature Engineering

Features are based on five dominant Fourier periods identified in solar cycle analysis:
- 65 months (5.45 years)
- 102 months (8.52 years)
- 131 months (10.91 years)
- 182 months (15.15 years)
- 654 months (54.53 years)

At time t, features are: `X(t) = [ssn(t-65), ssn(t-102), ssn(t-131), ssn(t-182), ssn(t-654)]`

### Train/Test Split

- **Test Set**: Fixed final 131 months (chronological, no shuffling)
- **Training Set**: All data before test window, optionally filtered by start year

### Cross-Validation

6-fold time-series cross-validation with temporal ordering:
- Each fold splits into 80% training, 20% validation
- Maintains chronological order within each fold
- Reports mean RMSE across validation folds

### Evaluation Protocol

For each (model, start_year) configuration:
1. Filter training data to start from specified year
2. Perform 6-fold time-series CV to compute validation RMSE
3. Retrain on full filtered training data
4. Evaluate on fixed 131-month test window
5. Report both CV RMSE and test RMSE

## Data

The `data/sunspots.csv` file contains SILSO V2.0 monthly sunspot data with columns:
- `year`, `month`: Date components
- `ssn_smoothed`: 13-month smoothed sunspot number
- `ssn_raw`: Raw monthly sunspot number
- `decimal_date`: Decimal year
- `provisional`, `revision`: Data quality indicators

Invalid values (≤ 0) in the first and last 6 months are automatically filtered.

## Adding New Models

To add a custom model:

1. Edit `src/models/registry.py`:
```python
def get_model(name: str):
    # ... existing code ...
    
    elif name == "my_model":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=100, random_state=42)
```

2. Update `AVAILABLE_MODELS` list

3. Run experiments:
```bash
python run_experiments.py --models my_model
```

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0

---

## Acknowledgments

### Data Source
This project uses monthly sunspot data from the **World Data Center SILSO** (Sunspot Index and Long-term Solar Observations), Royal Observatory of Belgium, Brussels.

**Citation:**
> SILSO, World Data Center - Sunspot Number and Long-term Solar Observations, Royal Observatory of Belgium, on-line Sunspot Number catalogue: http://www.sidc.be/SILSO/

### Methodology
The feature engineering approach and experimental design are based on prior research in solar cycle prediction using Fourier-based lag features and machine learning methods.

### Development
This codebase was developed with AI assistance as an educational and research tool for exploring time-series forecasting methods applied to solar activity data.

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

This software is provided as-is for research and educational purposes. No warranties or claims are made regarding the accuracy of predictions or suitability for any particular purpose.
