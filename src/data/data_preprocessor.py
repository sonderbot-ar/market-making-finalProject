# data_preprocessor.py

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
)


# ============================================================
# CONFIGURATION
# ============================================================

ROLLING_WINDOW = 60  # 60-minute rolling volatility

# INPUT_FILE = "src/data/test-BTCUSD_Bitstamp_1min_2025-06-12.csv"

# ============================================================
# FEATURE ENGINEERING FOR Avellaneda-Stoikov
# ============================================================

def compute_mid_price(df: pd.DataFrame) -> pd.Series:
    """
    Mid-price proxy for OHLC data.

    Option 1:
        S_t = Close_t

    Option 2 (commented):
        S_t = (High + Low + Close) / 3
    """
    return df["Close"]

    # Alternative:
    # return (df["High"] + df["Low"] + df["Close"]) / 3


def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """
    r_t = ln(S_t / S_{t-1})
    """
    return np.log(
        df["mid_price"] / df["mid_price"].shift(1)
    )


def compute_rolling_volatility(
    price_diff: pd.Series,
    window: int = ROLLING_WINDOW
) -> pd.Series:
    """
    Rolling standard deviation of price differences.

    σ_period = rolling std(ΔS)
    """
    return price_diff.rolling(window=window).std()


def compute_time_normalized_volatility(
    rolling_sigma: pd.Series
) -> pd.Series:
    """
    Convert minute volatility into per-second volatility.

    σ = σ_period / sqrt(60)
    """
    return rolling_sigma / np.sqrt(60)


def compute_fill_distances(df: pd.DataFrame):
    """
    δ^a = High_t - S_{t-1}
    δ^b = S_{t-1} - Low_t
    """

    prev_mid = df["mid_price"].shift(1)

    delta_ask = df["High"] - prev_mid
    delta_bid = prev_mid - df["Low"]

    # Prevent negative penetration distances
    delta_ask = delta_ask.clip(lower=0)
    delta_bid = delta_bid.clip(lower=0)

    return delta_ask, delta_bid


def compute_rolling_kappa(
    df: pd.DataFrame,
    window: int = 60
) -> pd.Series:
    """
    κ_t = 1 / mean(δ)
    """

    mean_delta = (
        (df["delta_ask"] + df["delta_bid"]) / 2
    ).rolling(window).mean()

    return 1.0 / mean_delta


def compute_rolling_A(
    df: pd.DataFrame,
    window: int = 60
) -> pd.Series:
    """
    A_t = fills / time
    """

    fill_events = (
        (df["delta_ask"] > 0).astype(int)
        +
        (df["delta_bid"] > 0).astype(int)
    )

    rolling_fills = fill_events.rolling(window).sum()

    rolling_seconds = window * 60

    return rolling_fills / rolling_seconds


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

def preprocess_dataset(
    input_file: str,
    rolling_window: int = ROLLING_WINDOW
):
    """
    Load CSV and generate AS features.
    """

    print(f"Loading: {input_file}")

    df = pd.read_csv(input_file)

    # --------------------------------------------------------
    # Mid Price
    # --------------------------------------------------------
    df["mid_price"] = compute_mid_price(df)

    # --------------------------------------------------------
    # Price Differences (Avellaneda-Stoikov)
    # --------------------------------------------------------
    df["price_diff"] = df["mid_price"].diff()

    # --------------------------------------------------------
    # Log Returns (optional ML feature)
    # --------------------------------------------------------
    df["log_return"] = compute_log_returns(df)

    # --------------------------------------------------------
    # Rolling Volatility
    # σ_period = std(ΔS)
    # --------------------------------------------------------
    df["rolling_stdev"] = compute_rolling_volatility(
        df["price_diff"],
        rolling_window
    )

    # --------------------------------------------------------
    # Time-Normalized Volatility
    # --------------------------------------------------------
    df["sigma"] = compute_time_normalized_volatility(
        df["rolling_stdev"]
    )

    # --------------------------------------------------------
    # Fill Distances
    # --------------------------------------------------------
    (
        df["delta_ask"],
        df["delta_bid"]
    ) = compute_fill_distances(df)

    # --------------------------------------------------------
    # Global Liquidity Parameters
    # --------------------------------------------------------
    
    df["kappa"] = compute_rolling_kappa(
    df,
    rolling_window
    )

    df["A"] = compute_rolling_A(
    df,
    rolling_window
    )

    return df

# Add another section here on feature engineering for another market making model
# >> insert here


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_processed_file(
    df: pd.DataFrame,
    input_file: str
):
    path = Path(input_file)

    output_file = (
        PROCESSED_DATA_DIR /
        f"{path.stem}_processed(AS).csv"
    )

    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    csv_files = list(
        RAW_DATA_DIR.glob("*.csv")
    )
    
    for file_path in csv_files:
        print(f"\nProcessing: {file_path.name}")

        df = preprocess_dataset(str(file_path))

        save_processed_file(df, str(file_path))

    print("\nAll files processed.")
    print("\nGenerated columns:")
    print(
                [
            "mid_price",
            "price_diff",
            "log_return",
            "rolling_stdev",
            "sigma",
            "delta_ask",
            "delta_bid",
            "kappa",
            "A",
        ]
    )