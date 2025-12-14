# NOTE: This Makefile was AI-generated and may require adjustments for your environment.

PYTHON = .venv/bin/python
PIP = .venv/bin/pip

.PHONY: help install verify quick train forecast training-fit rmse publish clean zip

help:
	@echo "Sunspot Prediction Experiment - Available Commands:"
	@echo ""
	@echo "MAIN WORKFLOW:"
	@echo "  make install      - Install dependencies"
	@echo "  make train        - Train all models (1804-1940, CV, merit weights)"
	@echo "  make publish      - Generate all publication figures"
	@echo ""
	@echo "MODULAR VISUALIZATION:"
	@echo "  make forecast     - Generate all model forecast plots"
	@echo "  make training-fit - Generate all model training fit plots"
	@echo "  make rmse         - Generate RMSE vs year plots (test & CV)"
	@echo ""
	@echo "UTILITY:"
	@echo "  make verify       - Verify data loading works"
	@echo "  make quick        - Quick test (2 models, few years, no CV)"
	@echo "  make clean        - Remove cache and temp files"
	@echo "  make zip          - Create archive of git-tracked files"
	@echo ""
	@echo "Default models (5): linear_regression, random_forest, xgboost, kneighbors, svm"

install:
	@if [ ! -d .venv ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv .venv; \
	fi
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "✓ Installation complete!"
	@echo "To activate the virtual environment, run:"
	@echo "  source .venv/bin/activate"

verify:
	$(PYTHON) src/data/sunspot_data.py

quick:
	$(PYTHON) main.py

train:
	@echo "Training all models (1804-1940, CV, merit weights)..."
	$(PYTHON) main.py --dense 1804 1940 --cv --merit-weights --mark-primary

forecast:
	@echo "Generating forecast plots for all models..."
	$(PYTHON) -m src.visualization.plot_forecast linear_regression 2020 2035
	$(PYTHON) -m src.visualization.plot_forecast random_forest 2020 2035
	$(PYTHON) -m src.visualization.plot_forecast xgboost 2020 2035
	$(PYTHON) -m src.visualization.plot_forecast kneighbors 2020 2035
	$(PYTHON) -m src.visualization.plot_forecast svm 2020 2035
	@echo "Forecast plots saved to figures/forecasts/"

training-fit:
	@echo "Generating training fit plots for all models (last 30 years)..."
	$(PYTHON) -c "from src.visualization.plot_training_fit import plot_training_fit; plot_training_fit('linear_regression', last_n_years=30, save=True, show=False)"
	$(PYTHON) -c "from src.visualization.plot_training_fit import plot_training_fit; plot_training_fit('random_forest', last_n_years=30, save=True, show=False)"
	$(PYTHON) -c "from src.visualization.plot_training_fit import plot_training_fit; plot_training_fit('xgboost', last_n_years=30, save=True, show=False)"
	$(PYTHON) -c "from src.visualization.plot_training_fit import plot_training_fit; plot_training_fit('kneighbors', last_n_years=30, save=True, show=False)"
	$(PYTHON) -c "from src.visualization.plot_training_fit import plot_training_fit; plot_training_fit('svm', last_n_years=30, save=True, show=False)"
	@echo "Training fit plots saved to figures/training_fits/"

rmse:
	@echo "Generating RMSE vs year plots (1804-1940)..."
	$(PYTHON) -m src.visualization.plot_rmse_vs_year results/experiment_results.csv --metric test --start-year 1804 --end-year 1940
	$(PYTHON) -m src.visualization.plot_rmse_vs_year results/experiment_results.csv --metric cv --start-year 1804 --end-year 1940
	@echo "RMSE plots saved to figures/rmse_analysis/"

publish:
	@echo "Generating all publication figures..."
	$(MAKE) forecast
	$(MAKE) training-fit
	$(MAKE) rmse
	@echo "Generating accuracy summary table..."
	$(PYTHON) -m src.visualization.plot_accuracy_summary
	@echo "Copying key figures to publish folder..."
	mkdir -p figures/publish
	cp figures/rmse_analysis/rmse_vs_year_test_1804_1940.png figures/publish/
	cp figures/rmse_analysis/rmse_vs_year_cv_1804_1940.png figures/publish/
	cp figures/rmse_analysis/accuracy_summary_table.png figures/publish/
	cp figures/forecasts/all_forecasts_2020_2035.png figures/publish/ 2>/dev/null || true
	@echo ""
	@echo "=== PUBLISH COMPLETE ==="
	@echo "All publication-ready figures are in figures/publish/"

clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf figures/forecasts figures/training_fits figures/rmse_analysis
	rm -rf results
	@echo "Cleaned! (figures/publish preserved)"

zip:
	@echo "Creating archive of git-tracked files..."
	git archive --format=zip --output=../aml-final-project.zip HEAD
	@echo "Saved: ../aml-final-project.zip"