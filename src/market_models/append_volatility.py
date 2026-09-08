import duckdb

def append_rolling_volatility(input_parquet, output_parquet):
    print(f"Calculating 60-second rolling volatility for {input_parquet}...")
    
    query = f"""
    COPY (
        SELECT 
            *,
            -- 1. Calculate mid price
            -- 2. Compute rolling standard deviation over a 60-second lookback
            COALESCE(
                STDDEV_SAMP((best_bid_spot + best_ask_spot) / 2.0) OVER (
                    ORDER BY timestamp 
                    RANGE BETWEEN INTERVAL 1 MINUTE PRECEDING AND CURRENT ROW
                ), 
                0.00001 -- Fallback for the very first few ticks to avoid zero division
            ) AS rolling_volatility
        FROM read_parquet('{input_parquet}')
    ) TO '{output_parquet}' (FORMAT PARQUET, COMPRESSION 'snappy');
    """
    
    duckdb.execute(query)
    print(f"Success! Dataset with volatility saved to {output_parquet}")

# --- TRIGGER LOGIC ---
append_rolling_volatility("/Users/andresrodartee/market-making-finalProject/data/data/merged/master_wfo_dataset.parquet", "master_wfo_ready.parquet")