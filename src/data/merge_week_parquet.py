import duckdb

def merge_spot_futures_duckdb(spot_path, futures_path, output_path):
    print("Initializing DuckDB out-of-core ASOF join...")
    print(f"Reading Spot: {spot_path}")
    print(f"Reading Futures: {futures_path}")
    
    # DuckDB will read the files directly from disk, perform the ASOF left join, 
    # and write the output directly to a new Parquet file.
    query = f"""
    COPY (
        SELECT 
            make_timestamp_ns(s.timestamp) AS timestamp,
            s.best_bid AS best_bid_spot,
            s.best_ask AS best_ask_spot,
            s.best_bid_qty AS best_bid_qty_spot,
            s.best_ask_qty AS best_ask_qty_spot,
            f.best_bid AS best_bid_futures,
            f.best_ask AS best_ask_futures,
            f.best_bid_qty AS best_bid_qty_futures,
            f.best_ask_qty AS best_ask_qty_futures
        FROM read_parquet('{spot_path}') s
        ASOF LEFT JOIN read_parquet('{futures_path}') f
          ON make_timestamp_ns(s.timestamp) >= epoch_ms(f.timestamp)
    ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION 'snappy');
    """
    
    # Execute the query
    duckdb.execute(query)
    print(f"Success! Master hedging dataset saved to {output_path}")

# --- TRIGGER LOGIC ---
spot_file = "/Users/andresrodartee/market-making-finalProject/data/data/merged/master_week_bbo.parquet"
futures_file = "/Users/andresrodartee/market-making-finalProject/data/data/merged/BTCUSDT_2024-02-08_to_2024-02-14_merged.parquet"
output_file = "/Users/andresrodartee/market-making-finalProject/data/data/merged/master_wfo_dataset.parquet"

merge_spot_futures_duckdb(spot_file, futures_file, output_file)