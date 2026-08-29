import pandas as pd

file_path = "/Users/andresrodartee/market-making-finalProject/data/BTCUSDT_2024-03-27_merged.parquet"

df = pd.read_parquet(file_path)

print(df.columns.to_list())

print("\nFirst row preview:")
print(df.head(1))

df.info()