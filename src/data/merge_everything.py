import duckdb

def merge_trades_to_master(master_path, trades_path, output_path):
    print("Snapping trade executions onto the master timeline...")
    
    query = f"""
    COPY (
        WITH grouped_trades AS (
            SELECT 
                timestamp,
                SUM(trade_qty) AS trade_qty,
                -- If multiple trades hit in the exact same nanosecond, 
                -- we take the mode (most common side) for the taker flag
                MODE(is_buyer_maker) AS is_buyer_maker
            FROM read_parquet('{trades_path}')
            GROUP BY timestamp
        )
        SELECT 
            m.*,
            t.trade_qty,
            t.is_buyer_maker
        FROM read_parquet('{master_path}') m
        LEFT JOIN grouped_trades t 
          ON m.timestamp = make_timestamp_ns(t.timestamp)
        ORDER BY m.timestamp
    ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION 'snappy');
    """
    
    duckdb.execute(query)
    print(f"Success! Final dataset with trades saved to {output_path}")

# --- TRIGGER LOGIC ---
merge_trades_to_master(
    "/Users/andresrodartee/market-making-finalProject/data/data/merged/master_wfo_ready.parquet", 
    "/Users/andresrodartee/market-making-finalProject/data/data/merged/spot_trades_merged.parquet", 
    "/Users/andresrodartee/market-making-finalProject/data/data/merged/master_wfo_with_trades.parquet"
)