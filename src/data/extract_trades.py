import zstandard as zstd
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import os

def extract_trades_from_folder(folder_path, output_parquet, chunk_size=500000):
    # Automatically find and sort all .txt.zst files in the folder
    search_pattern = os.path.join(folder_path, "*.txt.zst")
    file_list = sorted(glob.glob(search_pattern))
    
    if not file_list:
        print(f"No .txt.zst files found in {folder_path}!")
        return

    print(f"Found {len(file_list)} files. Extracting trades (Message Type 2)...\n")
    
    schema = pa.schema([
        ('timestamp', pa.int64()),
        ('trade_qty', pa.float64()),
        ('is_buyer_maker', pa.bool_())
    ])
    
    writer = pq.ParquetWriter(output_parquet, schema, compression='snappy')
    records = []
    
    for input_zst in file_list:
        print(f"Processing: {os.path.basename(input_zst)}")
        with open(input_zst, 'rb') as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                buffer = ""
                
                while True:
                    chunk = reader.read(65536).decode('utf-8')
                    if not chunk:
                        break
                    
                    buffer += chunk
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if not line.strip():
                            continue
                            
                        try:
                            data = json.loads(line)
                            
                            # Filter strictly for Message Type 2 (Trades)
                            if data[0] == 2:
                                records.append({
                                    'timestamp': data[4],
                                    'trade_qty': float(data[6][0][2]),
                                    'is_buyer_maker': bool(data[6][0][0])
                                })
                                
                                if len(records) >= chunk_size:
                                    df = pd.DataFrame(records)
                                    table = pa.Table.from_pandas(df, schema=schema)
                                    writer.write_table(table)
                                    records = []
                                    
                        except Exception:
                            continue

    if records:
        df = pd.DataFrame(records)
        table = pa.Table.from_pandas(df, schema=schema)
        writer.write_table(table)
            
    writer.close()
    print(f"\nSuccess! All trades extracted and saved to {output_parquet}")

# --- TRIGGER LOGIC ---
# Just point this to the folder containing your 7 files
extract_trades_from_folder("/Users/andresrodartee/market-making-finalProject/data/data/raw", "spot_trades_merged.parquet")