from src.avellaneda_stoikov import main as run_avellaneda_stoikov
from src.trained_model import main as run_xgboost_model
from src.backtest_version1 import main as run_backtest


def main():
    print("\nRunning Avellaneda-Stoikov benchmark...")
    run_avellaneda_stoikov()

    print("\nTraining and logging XGBoost model with MLflow...")
    run_xgboost_model()

    print("\nRunning backtest comparison...")
    run_backtest()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()