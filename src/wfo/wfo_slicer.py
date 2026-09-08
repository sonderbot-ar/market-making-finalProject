import duckdb
import pandas as pd
from datetime import timedelta

def generate_wfo_windows(parquet_path, train_hours=24, test_hours=3, step_hours=3):
    """
    A memory-safe generator that yields training and testing DataFrames 
    for Walk-Forward Optimization using DuckDB.
    """
    print(f"Initializing WFO Slicer for: {parquet_path}")
    con = duckdb.connect()
    
    # 1. Dynamically find the absolute start and end timestamps of the dataset
    query_bounds = f"""
        SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
        FROM read_parquet('{parquet_path}')
    """
    bounds = con.execute(query_bounds).df()
    min_ts = bounds['min_ts'].iloc[0]
    max_ts = bounds['max_ts'].iloc[0]
    
    print(f"Dataset boundaries detected: {min_ts} to {max_ts}\n")
    
    # 2. Initialize rolling window pointers
    current_train_start = min_ts
    wfo_step = 1
    
    while True:
        current_train_end = current_train_start + timedelta(hours=train_hours)
        current_test_end = current_train_end + timedelta(hours=test_hours)
        
        # Stop the generator if the test window exceeds available data
        if current_test_end > max_ts:
            print("WFO Slicer has reached the end of the dataset. Optimization complete.")
            break
            
        print(f"--- WFO Step {wfo_step} ---")
        print(f"Train: {current_train_start} -> {current_train_end}")
        print(f"Test:  {current_train_end} -> {current_test_end}")
        
        # 3. Extract the Train Window directly into a Pandas DataFrame
        query_train = f"""
            SELECT * FROM read_parquet('{parquet_path}')
            WHERE timestamp >= '{current_train_start}' 
              AND timestamp < '{current_train_end}'
            ORDER BY timestamp
        """
        df_train = con.execute(query_train).df()
        
        # 4. Extract the Test Window directly into a Pandas DataFrame
        query_test = f"""
            SELECT * FROM read_parquet('{parquet_path}')
            WHERE timestamp >= '{current_train_end}' 
              AND timestamp < '{current_test_end}'
            ORDER BY timestamp
        """
        df_test = con.execute(query_test).df()
        
        # Yield the DataFrames and metadata back to the main optimization loop
        yield {
            'wfo_step': wfo_step,
            'train_start_datetime': current_train_start,
            'train_end_datetime': current_train_end,
            'test_start_datetime': current_train_end,
            'test_end_datetime': current_test_end,
            'train_rows': len(df_train),
            'test_rows': len(df_test),
            'df_train': df_train,
            'df_test': df_test
        }
        
        # Roll the starting pointer forward by the defined step size (6 hours)
        current_train_start += timedelta(hours=step_hours)
        wfo_step += 1

# --- TEST LOGIC ---
# Uncomment this to test if your slices generate correctly before building the backtester
for window in generate_wfo_windows("/Users/andresrodartee/market-making-finalProject/data/data/merged/master_wfo_dataset.parquet"):
    print(f"Loaded {window['train_rows']} Train rows and {window['test_rows']} Test rows.\n")
    if window['wfo_step'] >= 3: # Stop after 3 slices to test
        break