import pandas as pd
import numpy as np
import pytest
from src.data.data_preprocessor import *

def test_compute_time_normalized_volatility():
    """
    Ensures the time-normalization math is strictly preserved.
    Formula: sigma = sigma_period / sqrt(60)
    """
    dummy_rolling_sigma = pd.Series([10.0, 60.0])
    
    result = compute_time_normalized_volatility(dummy_rolling_sigma)
    
    expected_1 = 10.0 / np.sqrt(60)
    expected_2 = 60.0 / np.sqrt(60)
    
    np.testing.assert_almost_equal(result.iloc[0], expected_1, decimal=4)
    np.testing.assert_almost_equal(result.iloc[1], expected_2, decimal=4)


def test_preprocess_dataset_contract(tmp_path):
    """
    Acts as a 'Smoke Test'. Creates a dummy CSV, runs it through 
    the full Avellaneda-Stoikov pipeline, and verifies the output shape.
    """
    dummy_data = pd.DataFrame({
        "High": [102.0, 103.0, 102.5],
        "Low": [100.0, 101.0, 100.5],
        "Close": [101.0, 102.0, 101.5]
    })
    
    dummy_file = tmp_path / "test_ticks.csv"
    dummy_data.to_csv(dummy_file, index=False)
    
    result_df = preprocess_dataset(str(dummy_file), rolling_window=2)
    
    expected_features = [
        "mid_price", "price_diff", "log_return", 
        "rolling_stdev", "sigma", "delta_ask", 
        "delta_bid", "kappa", "A"
    ]
    
    for feature in expected_features:
        assert feature in result_df.columns, f"Pipeline failed to generate critical AS feature: {feature}"
        
    pd.testing.assert_series_equal(
        result_df["mid_price"], 
        result_df["Close"], 
        check_names=False
    )