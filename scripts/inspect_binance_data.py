import zipfile
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

# Extract all zip files in data/raw
for zip_path in RAW_DIR.glob("*.zip"):
    print(f"Extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)

# Inspect extracted CSV files
for csv_path in RAW_DIR.glob("*.csv"):
    print("\n" + "=" * 80)
    print(csv_path.name)

    df = pd.read_csv(csv_path, nrows=5)
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())