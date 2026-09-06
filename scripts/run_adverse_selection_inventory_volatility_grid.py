from concurrent.futures import ProcessPoolExecutor
from itertools import product
import os
from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.grid_calibration import calibration_arrays, run_improved_summary


DATA_PATH = Path("data/processed/BTCUSDT_2024-03-27_merged.parquet")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FULL_DAY_CALIBRATION = True
CALIBRATION_SAMPLE_SIZE = None
DEBUG_SAMPLE_SIZE = 50_000
N_WORKERS = max(1, (os.cpu_count() or 1) - 1)

VOLATILITY_MULTIPLIERS = [75.0, 125.0, 150.0]
INVENTORY_SKEWS = [500.0, 750.0, 1000.0]
TREND_MULTIPLIERS = [0.0, 0.25]
TREND_SKEW_CAP_RATIOS = [0.25]
MIN_SPREAD_SAFETY_MULTIPLIERS = [1.2, 1.5]
VOLATILITY_PAUSE_QUANTILES = [0.95, 0.99]
LIQUIDATION_AGGRESSIVENESS_VALUES = [0.5]
LIQUIDATION_THRESHOLD = 0.8
MAX_INVENTORY = 0.005

_arrays = None
_volatility_cutoffs = None


def _sample_settings():
    if FULL_DAY_CALIBRATION:
        return None, "calibration_full_day_2024_03_27"
    if CALIBRATION_SAMPLE_SIZE is not None:
        return CALIBRATION_SAMPLE_SIZE, f"calibration_{CALIBRATION_SAMPLE_SIZE}_events"
    return DEBUG_SAMPLE_SIZE, "debug_50000_events"


def _load_sample(sample_size):
    data = pd.read_parquet(DATA_PATH)
    sample = data.copy() if sample_size is None else data.head(sample_size).copy()
    if "rolling_return" not in sample.columns:
        sample["rolling_return"] = sample["mid_price"].pct_change(1000).fillna(0)
    return data, sample


def _initialize_worker(sample_size):
    global _arrays, _volatility_cutoffs
    _, sample = _load_sample(sample_size)
    _arrays = calibration_arrays(sample)
    _volatility_cutoffs = {
        quantile: sample["rolling_volatility"].quantile(quantile)
        for quantile in VOLATILITY_PAUSE_QUANTILES
    }


def _run_configuration(params):
    (
        volatility_multiplier,
        inventory_skew,
        trend_multiplier,
        trend_skew_cap_ratio,
        min_spread_safety_multiplier,
        volatility_pause_quantile,
        liquidation_aggressiveness,
    ) = params
    summary = run_improved_summary(
        _arrays,
        volatility_multiplier=volatility_multiplier,
        inventory_skew=inventory_skew,
        trend_multiplier=trend_multiplier,
        trend_skew_cap_ratio=trend_skew_cap_ratio,
        min_spread_safety_multiplier=min_spread_safety_multiplier,
        max_allowed_volatility=_volatility_cutoffs[volatility_pause_quantile],
        liquidation_aggressiveness=liquidation_aggressiveness,
        liquidation_threshold=LIQUIDATION_THRESHOLD,
        max_inventory=MAX_INVENTORY,
    )
    return {
        **summary,
        "volatility_multiplier": volatility_multiplier,
        "inventory_skew": inventory_skew,
        "trend_multiplier": trend_multiplier,
        "trend_skew_cap_ratio": trend_skew_cap_ratio,
        "min_spread_safety_multiplier": min_spread_safety_multiplier,
        "volatility_pause_quantile": volatility_pause_quantile,
        "liquidation_aggressiveness": liquidation_aggressiveness,
        "liquidation_threshold": LIQUIDATION_THRESHOLD,
        "max_inventory": MAX_INVENTORY,
    }


def _run_grid(parameter_grid, sample_size):
    try:
        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            initializer=_initialize_worker,
            initargs=(sample_size,),
        ) as executor:
            return list(executor.map(_run_configuration, parameter_grid))
    except Exception as error:
        print(f"Parallel execution failed ({error}); falling back to sequential.")
        _initialize_worker(sample_size)
        return list(map(_run_configuration, parameter_grid))


def _ranking_columns():
    return [
        "volatility_multiplier",
        "inventory_skew",
        "trend_multiplier",
        "trend_skew_cap_ratio",
        "min_spread_safety_multiplier",
        "volatility_pause_quantile",
        "liquidation_aggressiveness",
        "final_net_pnl",
        "max_drawdown",
        "risk_adjusted_score",
        "fees_paid",
        "total_fills",
        "ending_inventory",
        "average_abs_inventory",
        "average_quoted_spread",
        "paused_events",
        "pause_rate",
        "long_liquidation_activations",
        "short_liquidation_activations",
    ]


def _print_rankings(grid_summary):
    columns = _ranking_columns()
    for title, metric in (
        ("final net P&L", "final_net_pnl"),
        ("max drawdown", "max_drawdown"),
        ("risk-adjusted score", "risk_adjusted_score"),
    ):
        print(f"\nTop 10 by {title}:")
        print(grid_summary.nlargest(10, metric)[columns].to_string(index=False))


def main():
    started_at = time.perf_counter()
    sample_size, run_label = _sample_settings()
    data, sample = _load_sample(sample_size)

    print("Original rows:", len(data))
    print("Rows used:", len(sample))
    print("First timestamp:", sample["timestamp"].iloc[0])
    print("Last timestamp:", sample["timestamp"].iloc[-1])
    print("Run label:", run_label)
    print("Full-day calibration:", FULL_DAY_CALIBRATION)

    parameter_grid = list(
        product(
            VOLATILITY_MULTIPLIERS,
            INVENTORY_SKEWS,
            TREND_MULTIPLIERS,
            TREND_SKEW_CAP_RATIOS,
            MIN_SPREAD_SAFETY_MULTIPLIERS,
            VOLATILITY_PAUSE_QUANTILES,
            LIQUIDATION_AGGRESSIVENESS_VALUES,
        )
    )
    print(f"Running {len(parameter_grid)} configurations with {N_WORKERS} workers...")
    summaries = _run_grid(parameter_grid, sample_size)

    grid_summary = pd.DataFrame(summaries)
    grid_summary.insert(0, "end_timestamp", sample["timestamp"].iloc[-1])
    grid_summary.insert(0, "start_timestamp", sample["timestamp"].iloc[0])
    grid_summary.insert(0, "sample_size", len(sample))
    grid_summary.insert(0, "run_label", run_label)
    grid_summary.insert(0, "model_version", "improved_calibration")
    grid_summary.insert(0, "model_name", "adverse_selection_inventory_volatility")
    grid_summary["risk_adjusted_score"] = (
        grid_summary["final_net_pnl"]
        + 0.5 * grid_summary["max_drawdown"]
        - 100 * grid_summary["average_abs_inventory"]
    )
    grid_summary = grid_summary.sort_values("final_net_pnl", ascending=False)

    output_path = RESULTS_DIR / (
        f"adverse_selection_inventory_volatility_grid_summary_{run_label}.csv"
    )
    grid_summary.to_csv(output_path, index=False)
    _print_rankings(grid_summary)

    comparison_columns = [
        "volatility_multiplier",
        "inventory_skew",
        "trend_multiplier",
        "trend_skew_cap_ratio",
        "min_spread_safety_multiplier",
        "volatility_pause_quantile",
        "liquidation_aggressiveness",
        "final_net_pnl",
        "max_drawdown",
        "risk_adjusted_score",
        "fees_paid",
        "total_fills",
        "average_abs_inventory",
        "ending_inventory",
    ]
    previous_selected = grid_summary.loc[
        (grid_summary["volatility_multiplier"] == 75.0)
        & (grid_summary["inventory_skew"] == 750.0)
        & (grid_summary["trend_multiplier"] == 0.0)
        & (grid_summary["trend_skew_cap_ratio"] == 0.25)
        & (grid_summary["min_spread_safety_multiplier"] == 1.5)
        & (grid_summary["volatility_pause_quantile"] == 0.95)
        & (grid_summary["liquidation_aggressiveness"] == 0.5)
    ]
    previous_pnl_winner = grid_summary.loc[
        (grid_summary["volatility_multiplier"] == 75.0)
        & (grid_summary["inventory_skew"] == 500.0)
        & (grid_summary["trend_multiplier"] == 0.0)
        & (grid_summary["trend_skew_cap_ratio"] == 0.25)
        & (grid_summary["min_spread_safety_multiplier"] == 1.5)
        & (grid_summary["volatility_pause_quantile"] == 0.99)
        & (grid_summary["liquidation_aggressiveness"] == 0.5)
    ]
    print("\nPrevious 500k selected configuration on the full day:")
    print(previous_selected[comparison_columns].to_string(index=False))
    print("\nPrevious 500k P&L winner on the full day:")
    print(previous_pnl_winner[comparison_columns].to_string(index=False))
    print(f"\nSaved {len(grid_summary)} runs to: {output_path}")
    print(f"Runtime seconds: {time.perf_counter() - started_at:.2f}")


if __name__ == "__main__":
    main()
