"""
Generate an accuracy summary table comparing models.
Shows Model, Best Start Year, Test RMSE, and Test Correlation.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data.sunspot_data import get_feature_matrix, LAGS
from src.models.registry import get_model

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "rmse_analysis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_accuracy_summary(
    csv_path: str | Path,
    save: bool = True,
    show: bool = False,
    output_filename: str | None = None,
) -> None:
    """
    Create a summary table showing best performance for each model.
    
    Table columns: Model, Best Start Year, Test RMSE, Test Corr.
    
    Args:
        csv_path: Path to experiment results CSV with best configurations
        save: Whether to save the figure
        show: Whether to display the figure
        output_filename: Custom output filename (optional)
    """
    # Load results to get the best configuration for each model
    df = pd.read_csv(csv_path)
    
    # For each model, find the configuration with lowest test RMSE
    best_configs = df.loc[df.groupby("model_name")["test_rmse"].idxmin()]
    
    # Now compute correlation for each model at its best configuration
    X, y = get_feature_matrix(LAGS)
    
    # Determine test set (last 131 months)
    test_size = 131
    X_test = X.iloc[-test_size:]
    y_test = y.iloc[-test_size:]
    
    model_stats = []
    
    for _, row in best_configs.iterrows():
        model_name = row["model_name"]
        start_year = row["start_year"]
        test_rmse = row["test_rmse"]
        
        # Filter training data by start year
        train_mask = (X.index.year >= start_year) & (X.index < X_test.index[0])  # type: ignore[attr-defined]
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        # Train model
        model = get_model(model_name)
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Compute correlation
        correlation = np.corrcoef(y_test, y_pred)[0, 1]
        
        model_stats.append({
            'model': model_name,
            'rmse': test_rmse,
            'correlation': correlation,
            'start_year': start_year
        })
        
        print(f"{model_name}: RMSE={test_rmse:.2f}, Corr={correlation:.4f}, Year={start_year}")
    
    # Sort alphabetically by model name
    model_stats_df = pd.DataFrame(model_stats)
    model_stats_df = model_stats_df.sort_values('model')
    
    # Create table data
    table_data = []
    for _, stats in model_stats_df.iterrows():
        model_label = stats['model'].replace('_', ' ').title()
        table_data.append([
            model_label,
            int(stats['start_year']),
            f"{stats['rmse']:.2f}",
            f"{stats['correlation']:.4f}"
        ])
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(8, len(table_data) * 0.6 + 1.2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'Best Start Year', 'Test RMSE', 'Test Corr.'],
        cellLoc='center',
        loc='center',
        colWidths=[0.28, 0.28, 0.22, 0.22]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)
    
    # Find best values
    all_rmse = [float(row[2]) for row in table_data]
    all_corr = [float(row[3]) for row in table_data]
    best_rmse = min(all_rmse)
    best_corr = max(all_corr)
    
    # Style data rows with alternating colors
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#FFFFFF')
            cell.set_edgecolor('black')
            cell.set_linewidth(1.5)
            
            # Make best RMSE bolder
            if j == 2:  # RMSE column
                rmse_val = float(table_data[i-1][2])
                if abs(rmse_val - best_rmse) < 0.01:
                    cell.set_text_props(weight='extra bold', color='darkgreen', fontsize=12)
            
            # Make best correlation bolder
            if j == 3:  # Correlation column
                corr_val = float(table_data[i-1][3])
                if abs(corr_val - best_corr) < 0.0001:
                    cell.set_text_props(weight='extra bold', color='darkgreen', fontsize=12)
    
    # Add title closer to table
    plt.suptitle(
        'Model Performance Summary',
        fontsize=14,
        fontweight='bold',
        y=0.80
    )
    
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    
    if save:
        if output_filename is None:
            output_filename = "accuracy_summary_table.png"
        output_path = FIGURES_DIR / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
        print(f"\nSaved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Use the most recent primary results
    results_dir = Path(__file__).resolve().parents[2] / "results"
    primary_results = results_dir / "experiment_results.csv"
    
    if primary_results.exists():
        plot_accuracy_summary(primary_results, save=True, show=False)
    else:
        print("No primary results found. Run experiments first.")
