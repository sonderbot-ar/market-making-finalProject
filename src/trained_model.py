import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import mlflow
import mlflow.xgboost

from src.config import (
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    MODELS_DIR
)

TRAIN_FILE = (
    PROCESSED_DATA_DIR /
    "train-BTCUSD_Bitstamp_1min_2024-06-14_processed(AS).csv"
)

VAL_FILE = (
    PROCESSED_DATA_DIR /
    "val-BTCUSD_Bitstamp_1min_2025-02-20_processed(AS).csv"
)

TEST_FILE = (
    PROCESSED_DATA_DIR /
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

def ensure_mid_price(df):
    df = df.copy()

    if "mid_price" not in df.columns:
        df["mid_price"] = (df["High"] + df["Low"]) / 2

    return df

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
        .shift(-1) - df["mid_price"]
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

    return {
        f"{name.lower()}_mae": mae,
        f"{name.lower()}_mse": mse,
        f"{name.lower()}_rmse": rmse,
        f"{name.lower()}_r2": r2
    }

def reconstruct_and_quote(df, preds, half_spread=10.0):
    # Convert predicted difference back into a raw price
    predicted_mid = df["mid_price"] + preds
    
    # Simulate market making quotes around the predicted mid
    bid = predicted_mid - half_spread
    ask = predicted_mid + half_spread
    
    return predicted_mid, bid, ask

def save_predictions(
    df,
    pred_mid,
    bid,
    ask,
    filename
):

    output = pd.DataFrame({
        "Datetime": df["Datetime"],
        "Current_MidPrice": df["mid_price"],
        "Actual_MidPrice": df["mid_price"].shift(-1),
        "Predicted_MidPrice": pred_mid,
        "bid": bid,
        "ask": ask
    })

    # Drop the final row which lacks a future Actual_MidPrice
    output = output.dropna()

    output.to_csv(
        filename,
        index=False
    )

def main():

    mlflow.set_experiment("market-making-xgboost")

    with mlflow.start_run() as run:

        train_df = ensure_mid_price(
        pd.read_csv(TRAIN_FILE)
        )

        val_df = ensure_mid_price(
        pd.read_csv(VAL_FILE)
        )

        test_df = ensure_mid_price(
        pd.read_csv(TEST_FILE)
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

        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("n_estimators", 500)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.03)
        mlflow.log_param("subsample", 0.8)
        mlflow.log_param("colsample_bytree", 0.8)
        mlflow.log_param("random_state", 42)

        model.fit(
            X_train,
            y_train
        )

        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            model,
            MODELS_DIR / "xgb_model.pkl"
        )

        val_pred_diff = model.predict(X_val)
        test_pred_diff = model.predict(X_test)

        val_pred_mid, val_bid, val_ask = reconstruct_and_quote(
            val_clean,
            val_pred_diff
        )

        test_pred_mid, test_bid, test_ask = reconstruct_and_quote(
            test_clean,
            test_pred_diff
        )

        val_actual_mid = val_clean["mid_price"] + y_val
        test_actual_mid = test_clean["mid_price"] + y_test

        val_metrics = evaluate(
            val_actual_mid,
            val_pred_mid,
            "Validation"
        )

        test_metrics = evaluate(
            test_actual_mid,
            test_pred_mid,
            "Test"
        )

        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)

        validation_predictions_path = (
            RESULTS_DIR / "xgb_validation_predictions.csv"
        )

        test_predictions_path = (
            RESULTS_DIR / "xgb_test_predictions.csv"
        )

        save_predictions(
            val_clean,
            val_pred_mid,
            val_bid,
            val_ask,
            validation_predictions_path
        )

        save_predictions(
            test_clean,
            test_pred_mid,
            test_bid,
            test_ask,
            test_predictions_path
        )

        mlflow.log_artifact(str(validation_predictions_path))
        mlflow.log_artifact(str(test_predictions_path))
        mlflow.log_artifact(str(MODELS_DIR / "xgb_model.pkl"))

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            input_example=X_train.head(1)
        )

        print("\nMLflow run completed.")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model URI: runs:/{run.info.run_id}/model")

if __name__ == "__main__":
    main()