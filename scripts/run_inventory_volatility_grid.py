from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.strategies.inventory_volatility import InventoryVolatilityMarketMaker


DATA_PATH = Path("data/processed/BTCUSDT_2024-03-27_merged.parquet")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

parameter_grid = [
    {
        "run_name": "vol25_skew250_inv0005",
        "volatility_multiplier": 25.0,
        "inventory_skew": 250.0,
        "max_inventory": 0.005,
    },
    {
        "run_name": "vol50_skew500_inv0005",
        "volatility_multiplier": 50.0,
        "inventory_skew": 500.0,
        "max_inventory": 0.005,
    },
    {
        "run_name": "vol100_skew500_inv0005",
        "volatility_multiplier": 100.0,
        "inventory_skew": 500.0,
        "max_inventory": 0.005,
    },
    {
        "run_name": "vol50_skew1000_inv0005",
        "volatility_multiplier": 50.0,
        "inventory_skew": 1000.0,
        "max_inventory": 0.005,
    },
    {
        "run_name": "vol50_skew500_inv0010",
        "volatility_multiplier": 50.0,
        "inventory_skew": 500.0,
        "max_inventory": 0.01,
    },
    {
        "run_name": "vol100_skew1000_inv0005",
        "volatility_multiplier": 100.0,
        "inventory_skew": 1000.0,
        "max_inventory": 0.005,
    },
]


def main():
    print("Loading data...")
    data = pd.read_parquet(DATA_PATH)
    data = data.head(50_000).copy()

    summaries = []

    for params in parameter_grid:
        print(f"Running {params['run_name']}...")
        strategy = InventoryVolatilityMarketMaker(
            base_spread_bps=0.0,
            volatility_multiplier=params["volatility_multiplier"],
            inventory_skew=params["inventory_skew"],
            order_size=0.001,
            max_inventory=params["max_inventory"],
            fee_rate=0.0002,
            starting_cash=100_000.0,
        )

        results = strategy.run_backtest(data)
        summary = strategy.summary(results)
        summary.update(
            {
                "run_name": params["run_name"],
                "volatility_multiplier": params["volatility_multiplier"],
                "inventory_skew": params["inventory_skew"],
                "max_inventory": params["max_inventory"],
            }
        )
        summaries.append(summary)

    grid_summary = pd.DataFrame(summaries).sort_values(
        "final_net_pnl", ascending=False
    )
    summary_path = RESULTS_DIR / "inventory_volatility_grid_summary.csv"
    grid_summary.to_csv(summary_path, index=False)

    print("\nGrid search complete.")
    print(grid_summary.to_string(index=False))
    print(f"\nSaved grid summary to: {summary_path}")


if __name__ == "__main__":
    main()
