import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


TRAIN_FILE = (
    "src/data/"
    "train-BTCUSD_Bitstamp_1min_2024-06-14_processed(AS).csv"
)

VAL_FILE = (
    "src/data/"
    "val-BTCUSD_Bitstamp_1min_2025-02-20_processed(AS).csv"
)

TEST_FILE = (
    "src/data/"
    "test-BTCUSD_Bitstamp_1min_2025-06-12_processed(AS).csv"
)


FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "mid_price"
]


def create_features(df):

    df = df.copy()

    df["return_1"] = (
        np.log(df["mid_price"])
        .diff()
    )

    df["return_5"] = (
        np.log(df["mid_price"])
        .diff(5)
    )

    df["ma_5"] = (
        df["mid_price"]
        .rolling(5)
        .mean()
    )

    df["ma_20"] = (
        df["mid_price"]
        .rolling(20)
        .mean()
    )

    df["volatility_20"] = (
        np.log(df["mid_price"])
        .diff()
        .rolling(20)
        .std()
    )

    df["target"] = (
        df["mid_price"]
        .shift(-1)
    )

    df = df.dropna()

    return df


def prepare(df):

    df = create_features(df)

    features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "return_1",
        "return_5",
        "ma_5",
        "ma_20",
        "volatility_20"
    ]

    X = df[features]

    y = df["target"]

    return X, y, df


def evaluate(
    y_true,
    y_pred,
    name
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        y_pred
    )

    print(f"\n{name}")
    print(f"MAE  = {mae:.8f}")
    print(f"MSE  = {mse:.8f}")
    print(f"RMSE = {rmse:.8f}")
    print(f"R²   = {r2:.8f}")


def save_predictions(
    df,
    preds,
    filename
):


    output = pd.DataFrame({
        "Datetime": df["Datetime"],
        "Actual_MidPrice": df["target"],
        "Predicted_MidPrice": preds
    })

    output.to_csv(
        filename,
        index=False
    )


def main():

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    val_df = pd.read_csv(
        VAL_FILE
    )

    test_df = pd.read_csv(
        TEST_FILE
    )

    X_train, y_train, train_clean = (
        prepare(train_df)
    )

    X_val, y_val, val_clean = (
        prepare(val_df)
    )

    X_test, y_test, test_clean = (
        prepare(test_df)
    )

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    val_pred = model.predict(
        X_val
    )

    test_pred = model.predict(
        X_test
    )

    evaluate(
        y_val,
        val_pred,
        "Validation"
    )

    evaluate(
        y_test,
        test_pred,
        "Test"
    )

    save_predictions(
        val_clean,
        val_pred,
        "src/results/validation_predictions.csv"
    )

    save_predictions(
        test_clean,
        test_pred,
        "src/results/test_predictions.csv"
    )


if __name__ == "__main__":
    main()