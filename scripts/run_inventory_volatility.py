from pathlib import Path
import pandas as pd

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.strategies.inventory_volatility import InventoryVolatilityMarketMaker


DATA_PATH = Path("data/processed/BTCUSDT_2024-03-27_merged.parquet")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading data...")
    data = pd.read_parquet(DATA_PATH)

    print("Original rows:", len(data))

    # Use a smaller sample first so it runs quickly
    data = data.head(50_000).copy()

    print("Sample rows:", len(data))
    print("Running inventory + volatility market maker...")

    strategy = InventoryVolatilityMarketMaker(
    base_spread_bps=0.0,
    volatility_multiplier=50.0,
    inventory_skew=500.0,
    order_size=0.001,
    max_inventory=0.005,
    fee_rate=0.0002,
    starting_cash=100_000.0,
)

    results = strategy.run_backtest(data)
    summary = strategy.summary(results)

    results_path = RESULTS_DIR / "inventory_volatility_results.csv"
    summary_path = RESULTS_DIR / "inventory_volatility_summary.csv"

    results.to_csv(results_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\nBacktest complete.")
    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nSaved results to: {results_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()