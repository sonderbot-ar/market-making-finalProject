import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

def auto_align_dataset(input_parquet, output_parquet, start_time="2024-02-08 00:00:00"):
    print("Loading master market state...")
    df = pd.read_parquet(input_parquet)
    
    time_col = None
    
    # 1. Check if the index is already a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        print("Detected time data in the DataFrame Index.")
        aligned_df = df[df.index >= start_time]
        
    else:
        # 2. Search for a datetime column automatically
        for col in df.columns:
            if is_datetime(df[col]) or col.lower() in ['timestamp', 'time', 'date', 'datetime']:
                time_col = col
                print(f"Detected time data in column: '{time_col}'")
                
                # Ensure it's properly formatted as datetime
                df[time_col] = pd.to_datetime(df[time_col])
                aligned_df = df[df[time_col] >= start_time]
                break
                
        if time_col is None:
            raise ValueError("Could not automatically detect a time column. Please check the parquet schema.")

    # 3. Verify the slice
    if time_col:
        print(f"Original start: {df[time_col].min()}")
        print(f"New aligned start: {aligned_df[time_col].min()}")
    else:
        print(f"Original start: {df.index.min()}")
        print(f"New aligned start: {aligned_df.index.min()}")
        
    # 4. Save for the PPO
    aligned_df.to_parquet(output_parquet)
    print(f"Successfully saved aligned dataset to {output_parquet}")

# Execute the alignment
auto_align_dataset(
    input_parquet="/Users/andresrodartee/market-making-finalProject/data/data/merged/master_wfo_final.parquet", 
    output_parquet="/Users/andresrodartee/market-making-finalProject/data/data/merged/btcusdt_ppo_aligned.parquet"
)