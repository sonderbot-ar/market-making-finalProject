import duckdb

def append_vpin_duckdb(input_parquet, output_parquet, bucket_volume=50.0, vpin_window=50):
    print("Calculating out-of-core VPIN using DuckDB...")
    
    query = f"""
    COPY (
        WITH raw_data AS (
            SELECT 
                *,
                COALESCE(trade_qty, 0.0) AS clean_trade_qty,
                COALESCE(is_buyer_maker, false) AS clean_is_buyer_maker
            FROM read_parquet('{input_parquet}')
        ),
        vol_calc AS (
            SELECT 
                *,
                CASE WHEN clean_is_buyer_maker THEN clean_trade_qty ELSE 0.0 END AS sell_vol,
                CASE WHEN NOT clean_is_buyer_maker THEN clean_trade_qty ELSE 0.0 END AS buy_vol,
                SUM(clean_trade_qty) OVER (ORDER BY timestamp) AS cumulative_vol
            FROM raw_data
        ),
        bucketed AS (
            SELECT 
                *,
                CAST(FLOOR(cumulative_vol / {bucket_volume}) AS INTEGER) AS bucket_id
            FROM vol_calc
        ),
        bucket_aggs AS (
            SELECT 
                bucket_id,
                SUM(buy_vol) AS buy_vol_sum,
                SUM(sell_vol) AS sell_vol_sum,
                ABS(SUM(buy_vol) - SUM(sell_vol)) AS imbalance
            FROM bucketed
            GROUP BY bucket_id
        ),
        rolling_vpin AS (
            SELECT 
                bucket_id,
                AVG(imbalance) OVER (
                    ORDER BY bucket_id 
                    ROWS BETWEEN {vpin_window - 1} PRECEDING AND CURRENT ROW
                ) / {bucket_volume} AS vpin
            FROM bucket_aggs
        )
        SELECT 
            b.*,
            COALESCE(r.vpin, 0.0) AS vpin
        FROM bucketed b
        LEFT JOIN rolling_vpin r ON b.bucket_id = r.bucket_id
        ORDER BY b.timestamp
    ) TO '{output_parquet}' (FORMAT PARQUET, COMPRESSION 'snappy');
    """
    
    duckdb.execute(query)
    print(f"Success! VPIN dataset saved to {output_parquet}")

# --- TRIGGER LOGIC ---
append_vpin_duckdb("/Users/andresrodartee/market-making-finalProject/data/data/merged/master_wfo_with_trades.parquet", "master_wfo_final.parquet")