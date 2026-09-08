import sys
import os

sys.path.append("/Users/andresrodartee/market-making-finalProject")

import pandas as pd
from wfo_engine import run_wfo_experiment
from src.market_models.vpin_hedge_model import simulate_vpin_hedge

def main():
    print("Initializing WFO Engine...")
    
    # 1. Define the M5-powered grid
    m5_param_grid = {
        'gamma': [0.01, 0.05, 0.1, 0.2],
        'kappa': [1.0, 1.5, 2.0],
        'A': [1.0],
        'hedge_threshold': [3, 5, 10, 15],
        'vpin_threshold': [0.6, 0.7, 0.8, 0.9]
    }
    
    # 2. Execute the Walk-Forward Optimization
    results_df, aggregate_summary = run_wfo_experiment(
        parquet_path="/Users/andresrodartee/market-making-finalProject/data/data/merged/master_wfo_final.parquet",
        model_name="VPIN_Delta_Hedge",
        model_version="v2.0",
        model_simulation_func=simulate_vpin_hedge,
        param_grid=m5_param_grid,
        fee_rate=0.0002,
        starting_cash=100_000.0
    )
    
    # 3. Save and display the results
    results_df.to_csv("/Users/andresrodartee/market-making-finalProject/results/wfo_vpin_results.csv", index=False)
    
    print("\n" + "="*50)
    print("WFO AGGREGATE SUMMARY")
    print("="*50)
    for key, value in aggregate_summary.items():
        print(f"{key}: {value}")
        
if __name__ == "__main__":
    main()