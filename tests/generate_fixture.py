import pandas as pd
import numpy as np
from pathlib import Path

def generate_dummy_data():
    """
    Generates a synthetic market data fixture for the CI/CD pipeline.
    Saves it to the data/raw/ directory to simulate real inputs.
    """
    target_dir = Path("data/raw")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({
        'High': np.random.uniform(101, 105, 100),
        'Low': np.random.uniform(95, 99, 100),
        'Close': np.random.uniform(99, 101, 100)
    })
    
    target_file = target_dir / "train-BTCUSD_Bitstamp_1min_2024-06-14.csv"
    df.to_csv(target_file, index=False)
    
    print(f"Dummy data successfully generated at: {target_file}")

if __name__ == "__main__":
    generate_dummy_data()