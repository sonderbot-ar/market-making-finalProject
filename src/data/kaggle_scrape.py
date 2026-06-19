import os
import kaggle
import pandas as pd

# How to use: just change the start and end dates according to necessity
# Plan:
# Training 1 - 2024-06-14 Post-ETF stable liquidity, ideal “clean market making environment”
# Training 2 - 2024-03-05 ETF hype + strong inflows + upward trend pressure
# Training 3 - 2024-08-05 Risk-off macro + crypto liquidation cascade behavior
# Validation 1 - 2025-02-20 Clean regime shift test
# Validation 2 - 2025-04-15 Slight expansion phase, tests parameter sensitivity
# Test 1 - 2025-06-12 ETF-driven stability period
# Test 2 - 2025-07-10 Tests robustness under tight spreads + sudden moves

DATASET_SLUG = "mczielinski/bitcoin-historical-data"
DOWNLOAD_DIR = "data"
start = "2025-07-10"
end = "2025-07-11"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

print("Downloading dataset...")
kaggle.api.dataset_download_files(
    DATASET_SLUG,
    path=DOWNLOAD_DIR,
    unzip=True
)

input_file = os.path.join(
    DOWNLOAD_DIR,
    "btcusd_1-min_data.csv"
)

print("Loading data...")
df = pd.read_csv(input_file)

df["Datetime"] = pd.to_datetime(
    df["Timestamp"],
    unit="s",
    utc=True
)

target_date = df[
    (df["Datetime"] >= start)
    & (df["Datetime"] < end)
]

output_file = os.path.join(
    DOWNLOAD_DIR,
    f"BTCUSD_Bitstamp_1min_{start}.csv"
)

target_date.to_csv(
    output_file,
    index=False
)

print(f"Saved {len(target_date)} rows")
print(f"File saved to: {output_file}")