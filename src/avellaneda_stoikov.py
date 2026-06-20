import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class AvellanedaStoikov:

    def __init__(self, gamma: float, sigma: float, kappa: float, T: float):
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa
        self.T = T

    def reservation_price(self, s: float, q: float, t: float) -> float:
        return s - q * self.gamma * (self.sigma ** 2) * (self.T - t)

    def optimal_total_spread(self, t: float) -> float:
        return (
            self.gamma * (self.sigma ** 2) * (self.T - t)
            + (2 / self.gamma) * np.log(1 + self.gamma / self.kappa)
        )

    def quote_prices(self, s: float, q: float, t: float):

        r = self.reservation_price(s, q, t)
        spread = self.optimal_total_spread(t)

        bid = r - spread / 2
        ask = r + spread / 2

        return r, bid, ask


# -----------------------------
# PIPELINE FUNCTIONS
# -----------------------------

def run_as_model(df, model, output_path):

    inventory = 1 # set as initial value
    results = []

    n = len(df)

    next_mid = (
        df["mid_price"]
        .shift(-1)
    )

    df = df.copy()
    df["next_mid"] = next_mid
    df = df.dropna()  

    for i, row in enumerate(df.itertuples(index=False)):

        t = i / n
        mid = row.mid_price

        r, bid, ask = model.quote_prices(mid, inventory, t)

        results.append({
            "Datetime": row.Datetime,
            "Current_MidPrice": mid,
            "Actual_MidPrice": row.next_mid,
            "Predicted_MidPrice": r,
            "bid": bid,
            "ask": ask
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)

    return out_df


def evaluate_as(df):

    mae = mean_absolute_error(
        df["Actual_MidPrice"],
        df["Predicted_MidPrice"]
    )

    mse = mean_squared_error(
        df["Actual_MidPrice"],
        df["Predicted_MidPrice"]
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        df["Actual_MidPrice"],
        df["Predicted_MidPrice"]
    )

    return mae, mse, rmse, r2


# -----------------------------
# MAIN ENTRY POINT (IMPORTANT)
# -----------------------------

def main():

    TRAIN_FILE = "src/data/train-BTCUSD_Bitstamp_1min_2024-06-14_processed(AS).csv"
    VAL_FILE = "src/data/val-BTCUSD_Bitstamp_1min_2025-02-20_processed(AS).csv"
    TEST_FILE = "src/data/test-BTCUSD_Bitstamp_1min_2025-06-12_processed(AS).csv"

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # simple parameter estimation
    sigma = train_df["mid_price"].pct_change().std()
    kappa = 1 / (train_df["mid_price"].diff().abs().mean() + 1e-9)

    model = AvellanedaStoikov(
        gamma=0.1,
        sigma=sigma,
        kappa=kappa,
        T=1.0
    )

    val_out = run_as_model(
        val_df,
        model,
        "src/results/as_validation.csv"
    )

    test_out = run_as_model(
        test_df,
        model,
        "src/results/as_test.csv"
    )

    val_mae, val_mse, val_rmse, val_r2 = evaluate_as(val_out)
    test_mae, test_mse, test_rmse, test_r2 = evaluate_as(test_out)

    print("\nAS MODEL RESULTS")

    print("\nValidation")
    print(f"MAE  = {val_mae:.8f}")
    print(f"MSE  = {val_mse:.8f}")
    print(f"RMSE = {val_rmse:.8f}")
    print(f"R²   = {val_r2:.8f}")

    print("\nTest")
    print(f"MAE  = {test_mae:.8f}")
    print(f"MSE  = {test_mse:.8f}")
    print(f"RMSE = {test_rmse:.8f}")
    print(f"R²   = {test_r2:.8f}")


if __name__ == "__main__":
    main()