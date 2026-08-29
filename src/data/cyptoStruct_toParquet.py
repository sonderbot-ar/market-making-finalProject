import io
import json
import pandas as pd
import zstandard as zstd

zstd_file_path = "/Users/andresrodartee/market-making-finalProject/binance_spot-BTCUSDT-2024-03-27.txt.zst"  # Update with your actual filename
parquet_out_path = "btcusdt_spot_bbo.parquet"

parsed_data = []

with open(zstd_file_path, "rb") as f:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(f, read_across_frames=True) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        
        for line in text_stream:
            line = line.strip()
            
            # Skip empty lines, JSON headers, and READY messages
            if not line or line.startswith('{') or '"READY"' in line:
                continue
            
            try:
                row = json.loads(line)
                
                # Filter ONLY for Message Type 6 (Top-of-Book / BBO)
                if row[0] == 6 and len(row) >= 7:
                    timestamp_ns = row[5]
                    payload = row[6]
                    
                    best_bid = None
                    best_bid_qty = None
                    best_ask = None
                    best_ask_qty = None
                    
                    # Iterate through the payload (which contains [side, price, qty, count])
                    for level in payload:
                        side = level[0]
                        price = float(level[1])
                        qty = float(level[2])
                        
                        if side == 0:  # Side 0 is Bid
                            best_bid = price
                            best_bid_qty = qty
                        elif side == 1:  # Side 1 is Ask
                            best_ask = price
                            best_ask_qty = qty
                            
                    # Ensure both Bid and Ask are present before appending
                    if best_bid is not None and best_ask is not None:
                        parsed_data.append({
                            "timestamp": timestamp_ns,
                            "best_bid": best_bid,
                            "best_bid_qty": best_bid_qty,
                            "best_ask": best_ask,
                            "best_ask_qty": best_ask_qty
                        })
                        
            except (json.JSONDecodeError, IndexError, ValueError):
                continue

print(f"Extraction complete. Found {len(parsed_data)} valid BBO updates.")

if len(parsed_data) == 0:
    print("CRITICAL: No data was extracted! Check the filepath.")
else:
    print("Building Parquet DataFrame...")
    df = pd.DataFrame(parsed_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
    df.to_parquet(parquet_out_path, compression="snappy")
    print(f"Successfully saved to {parquet_out_path}")