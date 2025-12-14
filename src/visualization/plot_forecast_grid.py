"""
Create vertical grid layouts of forecast plots.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "publish"
FORECASTS_DIR = Path(__file__).resolve().parents[2] / "figures" / "forecasts"


def plot_forecast_grid(
    models: list[str] | None = None,
    rows: int | None = None,
    cols: int = 1,
    save: bool = True,
    show: bool = False,
    output_filename: str | None = None,
) -> None:
    """
    Create a grid of forecast plots.
    
    Args:
        models: List of model names (default: ['linear_regression', 'kneighbors', 'xgboost'])
        rows: Number of rows (default: auto-calculated from number of models)
        cols: Number of columns (default: 1)
        save: Whether to save the figure
        show: Whether to display the figure
        output_filename: Name of output file (default: forecasts_grid_2020_2035.png)
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Default to 3 models
    if models is None:
        models = ['linear_regression', 'kneighbors', 'xgboost']
    
    # Auto-calculate rows if not specified
    if rows is None:
        rows = len(models)
    
    # Load forecast images
    images = []
    for model in models:
        img_path = FORECASTS_DIR / f'forecast_{model}_2020_2035.png'
        if img_path.exists():
            images.append(mpimg.imread(img_path))
        else:
            print(f"Warning: {img_path} not found")
            return
    
    # Create grid
    fig, axes = plt.subplots(rows, cols, figsize=(14 * cols, 4 * rows))
    fig.suptitle('Sunspot Forecasts', fontsize=18, fontweight='bold', y=0.995)
    
    # Handle single subplot case
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each image
    for idx, img in enumerate(images):
        if idx < len(axes):
            axes[idx].imshow(img)
            axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')
    
    # Adjust layout
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    
    if save:
        if output_filename is None:
            output_filename = "forecasts_grid_2020_2035.png"
        output_path = FIGURES_DIR / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {output_path}')
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create grid of forecast plots")
    parser.add_argument("--models", nargs="+", 
                        default=['linear_regression', 'kneighbors', 'xgboost'],
                        help="Model names to include (default: linear_regression kneighbors xgboost)")
    parser.add_argument("--rows", type=int, help="Number of rows (default: auto)")
    parser.add_argument("--cols", type=int, default=1, help="Number of columns (default: 1)")
    parser.add_argument("--output", help="Custom output filename")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    
    args = parser.parse_args()
    
    plot_forecast_grid(
        models=args.models,
        rows=args.rows,
        cols=args.cols,
        save=True,
        show=args.show,
        output_filename=args.output,
    )
