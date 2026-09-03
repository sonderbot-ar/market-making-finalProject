from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.inventory_volatility import InventoryVolatilityMarketMaker


DATA_PATH = Path("data/processed/BTCUSDT_2024-03-27_merged.parquet")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_FULL_DAY = True
SAMPLE_SIZE = 50_000
SAVE_FULL_EVENT_LOG = False
DOWNSAMPLE_EVERY = 1000


def main():
    print("Loading data...")
    data = pd.read_parquet(DATA_PATH)

    if RUN_FULL_DAY:
        backtest_data = data.copy()
        run_label = "full_day"
        run_mode = "full day"
    else:
        backtest_data = data.head(SAMPLE_SIZE).copy()
        run_label = f"first_{SAMPLE_SIZE}_events"
        run_mode = "sample"

    print("Original rows:", len(data))
    print("Rows used in this run:", len(backtest_data))
    print("First timestamp:", backtest_data["timestamp"].iloc[0])
    print("Last timestamp:", backtest_data["timestamp"].iloc[-1])
    print("Run mode:", run_mode)
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

    results = strategy.run_backtest(backtest_data)
    strategy_summary = strategy.summary(results)
    summary = {
        "model_name": "inventory_volatility",
        "model_version": "base",
        "run_label": run_label,
        "sample_size": len(backtest_data),
        "start_timestamp": backtest_data["timestamp"].iloc[0],
        "end_timestamp": backtest_data["timestamp"].iloc[-1],
        **strategy_summary,
    }

    summary_path = RESULTS_DIR / f"inventory_volatility_summary_{run_label}.csv"
    trades_path = RESULTS_DIR / f"inventory_volatility_trades_{run_label}.csv"
    plot_path = RESULTS_DIR / f"inventory_volatility_plot_{run_label}.csv"

    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    trades = results.loc[results["bid_filled"] | results["ask_filled"]]
    trades.to_csv(trades_path, index=False)
    results.iloc[::DOWNSAMPLE_EVERY].to_csv(plot_path, index=False)

    if SAVE_FULL_EVENT_LOG:
        results_path = RESULTS_DIR / f"inventory_volatility_results_{run_label}.parquet"
        results.to_parquet(results_path, index=False)

    print("\nBacktest complete.")
    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"Saved summary to: {summary_path}")
    print(f"Saved trades to: {trades_path}")
    print(f"Saved plot data to: {plot_path}")
    if SAVE_FULL_EVENT_LOG:
        print(f"Saved full event log to: {results_path}")


if __name__ == "__main__":
    main()
