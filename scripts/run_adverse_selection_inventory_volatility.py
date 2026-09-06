from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.adverse_selection_inventory_volatility import (
    AdverseSelectionAwareInventoryVolatilityMarketMaker,
)


DATA_PATH = Path("data/processed/BTCUSDT_2024-03-27_merged.parquet")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_FULL_DAY = True
SAMPLE_SIZE = 50_000
SAVE_FULL_EVENT_LOG = False
DOWNSAMPLE_EVERY = 1000
VOLATILITY_PAUSE_QUANTILE = 0.99


def main():
    print("Loading data...")
    data = pd.read_parquet(DATA_PATH)

    if RUN_FULL_DAY:
        backtest_data = data.copy()
        run_label = "calibration_full_day_2024_03_27_improved_selected"
        run_mode = "full day"
    else:
        backtest_data = data.head(SAMPLE_SIZE).copy()
        run_label = f"first_{SAMPLE_SIZE}_events"
        run_mode = "sample"

    if "rolling_return" not in backtest_data.columns:
        backtest_data["rolling_return"] = (
            backtest_data["mid_price"].pct_change(1000).fillna(0)
        )

    max_allowed_volatility = None
    if VOLATILITY_PAUSE_QUANTILE is not None:
        max_allowed_volatility = backtest_data["rolling_volatility"].quantile(
            VOLATILITY_PAUSE_QUANTILE
        )

    print("Original rows:", len(data))
    print("Rows used in this run:", len(backtest_data))
    print("First timestamp:", backtest_data["timestamp"].iloc[0])
    print("Last timestamp:", backtest_data["timestamp"].iloc[-1])
    print("Run mode:", run_mode)
    print("Maximum allowed volatility:", max_allowed_volatility)
    print("Running adverse-selection-aware market maker...")

    # Selected from full-day calibration on 2024-03-27. This calibration day is
    # in-sample; final evaluation will be performed on a separate out-of-sample test week.
    strategy = AdverseSelectionAwareInventoryVolatilityMarketMaker(
        base_spread_bps=0.0,
        volatility_multiplier=75.0,
        inventory_skew=500.0,
        trend_multiplier=0.0,
        trend_skew_cap_ratio=0.25,
        min_spread_safety_multiplier=1.5,
        liquidation_threshold=0.8,
        liquidation_aggressiveness=0.5,
        max_allowed_volatility=max_allowed_volatility,
        order_size=0.001,
        max_inventory=0.005,
        fee_rate=0.0002,
        starting_cash=100_000.0,
    )

    results = strategy.run_backtest(backtest_data)
    strategy_summary = strategy.summary(results)
    summary = {
        "model_name": "adverse_selection_inventory_volatility",
        "model_version": "improved",
        "run_label": run_label,
        "sample_size": len(backtest_data),
        "start_timestamp": backtest_data["timestamp"].iloc[0],
        "end_timestamp": backtest_data["timestamp"].iloc[-1],
        **strategy_summary,
    }
    summary["risk_adjusted_score"] = (
        summary["final_net_pnl"]
        + 0.5 * summary["max_drawdown"]
        - 100 * summary["average_abs_inventory"]
    )

    summary_path = (
        RESULTS_DIR / f"adverse_selection_inventory_volatility_summary_{run_label}.csv"
    )
    trades_path = (
        RESULTS_DIR / f"adverse_selection_inventory_volatility_trades_{run_label}.csv"
    )
    plot_path = (
        RESULTS_DIR / f"adverse_selection_inventory_volatility_plot_{run_label}.csv"
    )

    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    trades = results.loc[results["bid_filled"] | results["ask_filled"]]
    trades.to_csv(trades_path, index=False)
    results.iloc[::DOWNSAMPLE_EVERY].to_csv(plot_path, index=False)

    if SAVE_FULL_EVENT_LOG:
        results_path = RESULTS_DIR / (
            f"adverse_selection_inventory_volatility_results_{run_label}.parquet"
        )
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
