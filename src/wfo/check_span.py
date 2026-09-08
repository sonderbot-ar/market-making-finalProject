import duckdb

def check_file_health(file_path):
    print(f"\n--- Checking: {file_path} ---")
    try:
        query = f"""
            SELECT 
                MIN(timestamp) as start_time,
                MAX(timestamp) as end_time,
                ANY_VALUE(typeof(timestamp)) as data_type
            FROM read_parquet('{file_path}')
        """
        df = duckdb.execute(query).df()
        print(df.to_string())
    except Exception as e:
        print(f"File not found or error: {e}")

# --- TRIGGER LOGIC ---
base_path = "/Users/andresrodartee/market-making-finalProject/data/data/merged/"

check_file_health(base_path + "master_week_bbo.parquet")         # Step 1: Extracted Spot
check_file_health(base_path + "master_wfo_dataset.parquet")      # Step 2: Spot + Futures Merge
check_file_health(base_path + "master_wfo_ready.parquet")        # Step 3: Rolling Volatility
check_file_health(base_path + "master_wfo_with_trades.parquet")  # Step 4: Trades Merged
check_file_health(base_path + "master_wfo_final.parquet")        # Step 5: VPIN Final