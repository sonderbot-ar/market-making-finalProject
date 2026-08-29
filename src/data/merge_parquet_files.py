import pandas as pd

def merge_market_data(spot_path, futures_path, output_path):
    print("Loading Parquet files...")
    spot_df = pd.read_parquet(spot_path)
    futures_df = pd.read_parquet(futures_path)
    
    print("Standardizing timestamps...")
    # Convert Futures int64 ms to datetime64[ns] to match Spot
    futures_df['timestamp'] = pd.to_datetime(futures_df['timestamp'], unit='ms')
    
    print("Sorting time series...")
    # merge_asof strictly requires both dataframes to be sorted chronologically
    spot_df = spot_df.sort_values('timestamp')
    futures_df = futures_df.sort_values('timestamp')
    
    print("Executing As-Of Merge...")
    # Merges the most recent Spot quote to the exact millisecond of the Futures event
    merged_df = pd.merge_asof(
        futures_df,
        spot_df,
        on='timestamp',
        direction='backward',
        suffixes=('_futures', '_spot')
    )
    
    # Drop rows at the beginning where Futures events occurred before the first Spot quote
    merged_df = merged_df.dropna(subset=['best_bid_spot', 'best_ask_spot'])
    
    print("Saving Master Market State...")
    merged_df.to_parquet(output_path, compression="snappy")
    print(f"Successfully saved {len(merged_df)} rows to {output_path}")

# To run:
merge_market_data("/Users/andresrodartee/market-making-finalProject/data/btcusdt_spot_bbo.parquet", "/Users/andresrodartee/market-making-finalProject/data/BTCUSDT_2024-03-27_merged.parquet", "btcusdt_master_merged.parquet")