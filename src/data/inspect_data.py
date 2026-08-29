import pandas as pd

def inspect_praquet(file_path):
    df = pd.read_parquet(file_path)

    df.info()

    print(df.head(5))


if __name__ == '__main__':
    file_path = "/Users/andresrodartee/market-making-finalProject/data/BTCUSDT_2024-03-27_merged.parquet"

    inspect_praquet(file_path)