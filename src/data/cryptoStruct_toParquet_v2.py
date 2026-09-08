import io
import json
import pandas as pd
import zstandard as zstd
import glob
import os

# --- Configuration ---
input_folder = "/Users/andresrodartee/market-making-finalProject/data/data/raw" # Folder with .txt.zst files
master_out_path = "master_week_bbo.parquet"

def parse_zst_to_parquet(zstd_file_path, output_parquet_path):
    parsed_data = []
    print(f"Extracting BBO from: {os.path.basename(zstd_file_path)}...")
    
    with open(zstd_file_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f, read_across_frames=True) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            
            for line in text_stream:
                line = line.strip()
                if not line or line.startswith('{') or '"READY"' in line:
                    continue
                
                try:
                    row = json.loads(line)
                    # Filter ONLY for Message Type 6 (Top-of-Book / BBO)
                    if row[0] == 6 and len(row) >= 7:
                        timestamp_ns = row[4]
                        payload = row[6]
                        
                        best_bid, best_bid_qty, best_ask, best_ask_qty = None, None, None, None
                        
                        for level in payload:
                            side = level[0]
                            price = float(level[1])
                            qty = float(level[2])
                            
                            if side == 0:
                                best_bid, best_bid_qty = price, qty
                            elif side == 1:
                                best_ask, best_ask_qty = price, qty
                                
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

    if len(parsed_data) > 0:
        df = pd.DataFrame(parsed_data)
        df['timestamp'] = df['timestamp'].astype('int64')
        df.to_parquet(output_parquet_path, compression="snappy")
        print(f"   -> Saved {len(df)} rows to {output_parquet_path}")
        return output_parquet_path
    else:
        print(f"   -> WARNING: No valid data found in {zstd_file_path}")
        return None

# --- Main Execution ---
zst_files = sorted(glob.glob(os.path.join(input_folder, "*.txt.zst")))
daily_parquets = []

# 1. Process each file individually to manage RAM
for file in zst_files:
    daily_out = file.replace(".txt.zst", ".parquet")
    result = parse_zst_to_parquet(file, daily_out)
    if result:
        daily_parquets.append(result)

# 2. Concatenate all daily files into the master dataset
print("\nConcatenating daily files into master dataset...")
df_list = [pd.read_parquet(p) for p in daily_parquets]
master_df = pd.concat(df_list, ignore_index=True)

# Sort by timestamp just in case the files overlapped
master_df = master_df.sort_values("timestamp").reset_index(drop=True)
master_df.to_parquet(master_out_path, compression="snappy")

print(f"Success! Master week dataset saved to {master_out_path} with {len(master_df)} total rows.")