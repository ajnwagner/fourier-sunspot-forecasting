#!/usr/bin/env python3
"""Generate forecast plots for all default models."""

import sys
sys.path.insert(0, '/home/ajnw/aml-final-project/src')

from visualization.plot_forecast import plot_forecast
from models.registry import DEFAULT_MODELS

def main():
    print("Generating forecasts for all default models...")
    print(f"Models: {DEFAULT_MODELS}")
    
    for model_name in DEFAULT_MODELS:
        print(f"\n📊 Generating forecast for {model_name}...")
        try:
            plot_forecast(
                model_name,
                years_past=15,
                years_future=15,
                save=True,
                show=False
            )
            print(f"✅ Completed {model_name}")
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
    
    print("\n✨ All forecasts complete!")

if __name__ == "__main__":
    main()
