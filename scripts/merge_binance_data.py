from pathlib import Path
import pandas as pd
import numpy as np


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DATE = "2024-03-27"
SYMBOL = "BTCUSDT"

agg_path = RAW_DIR / f"{SYMBOL}-aggTrades-{DATE}.csv"
book_path = RAW_DIR / f"{SYMBOL}-bookTicker-{DATE}.csv"

print("Reading aggTrades...")
trades = pd.read_csv(agg_path)

print("Reading bookTicker...")
book = pd.read_csv(book_path)

# Rename trade columns
trades = trades.rename(
    columns={
        "price": "trade_price",
        "quantity": "trade_qty",
        "transact_time": "timestamp",
    }
)

# Rename book columns
book = book.rename(
    columns={
        "transaction_time": "timestamp",
        "best_bid_price": "best_bid",
        "best_bid_qty": "best_bid_qty",
        "best_ask_price": "best_ask",
        "best_ask_qty": "best_ask_qty",
    }
)

# Keep only useful columns
trades = trades[
    [
        "timestamp",
        "agg_trade_id",
        "trade_price",
        "trade_qty",
        "is_buyer_maker",
    ]
].copy()

book = book[
    [
        "timestamp",
        "best_bid",
        "best_bid_qty",
        "best_ask",
        "best_ask_qty",
    ]
].copy()

# Sort for asof merge
trades = trades.sort_values("timestamp")
book = book.sort_values("timestamp")

print("Merging trades with latest bookTicker quote...")
merged = pd.merge_asof(
    trades,
    book,
    on="timestamp",
    direction="backward",
)

# Drop rows where no quote was available before the trade
merged = merged.dropna(subset=["best_bid", "best_ask"])

# Add useful market-making fields
merged["mid_price"] = (merged["best_bid"] + merged["best_ask"]) / 2
merged["market_spread"] = merged["best_ask"] - merged["best_bid"]

# Trade side interpretation:
# Binance is_buyer_maker = True means buyer was maker, so aggressor was seller.
# False means seller was maker, so aggressor was buyer.
merged["aggressor_side"] = merged["is_buyer_maker"].map(
    {
        True: "sell",
        False: "buy",
    }
)

# Returns and rolling volatility
merged["log_return"] = np.log(merged["mid_price"]).diff()

# Rolling volatility over last 1,000 trade events
merged["rolling_volatility"] = merged["log_return"].rolling(window=1000).std()

# Remove first rows without volatility
merged = merged.dropna(subset=["rolling_volatility"])

out_path = PROCESSED_DIR / f"{SYMBOL}_{DATE}_merged.parquet"
merged.to_parquet(out_path, index=False)

print(f"Saved merged dataset to: {out_path}")
print("Rows:", len(merged))
print("Columns:")
print(merged.columns.tolist())
print(merged.head())