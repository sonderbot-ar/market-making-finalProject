import pandas as pd
import numpy as np

def run_backtest(file_path, strategy_name, trade_size=0.1):
    """
    Simulates market making by stepping through historical predictions.
    trade_size: The amount of BTC to buy/sell per fill.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Make sure the models have run.")
        return

    # Backtester State Variables
    cash = 0.0
    inventory = 0.0
    fills = 0
    
    portfolio_values = []
    inventory_history = []

    for idx, row in df.iterrows():
        bid = row['bid']
        ask = row['ask']
        next_price = row['Actual_MidPrice']

        # 1. Simulate Fills
        # If the market price falls to or below our bid, our buy order is filled.
        if next_price <= bid:
            inventory += trade_size
            cash -= bid * trade_size
            fills += 1
            
        # If the market price rises to or above our ask, our sell order is filled.
        elif next_price >= ask:
            inventory -= trade_size
            cash += ask * trade_size
            fills += 1

        # 2. Mark-to-Market Portfolio Valuation
        # Total value = the cash we hold + the current market value of our BTC inventory
        current_portfolio_value = cash + (inventory * next_price)
        
        portfolio_values.append(current_portfolio_value)
        inventory_history.append(inventory)

    # 3. Calculate Performance Metrics
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) # Minute-by-minute changes in wealth
    
    final_pnl = portfolio_values[-1]
    max_inventory_exposure = max(max(inventory_history), abs(min(inventory_history)))
    
    # Calculate Sharpe Ratio (Annualized for 1-minute data)
    # 365 days * 24 hours * 60 minutes = 525,600 minutes per year
    if len(returns) > 0 and np.std(returns) != 0:
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(525600)
    else:
        sharpe_ratio = 0.0

    # 4. Print Results
    print(f"\n" + "="*30)
    print(f" RESULTS: {strategy_name.upper()}")
    print(f" " + "="*29)
    print(f" Total Fills     : {fills}")
    print(f" Max Inv Exposure: {max_inventory_exposure:.2f} BTC")
    print(f" Final PnL       : ${final_pnl:.2f}")
    print(f" Sharpe Ratio    : {sharpe_ratio:.2f}")
    print(f"="*30)


def main():
    
    # 1. Evaluate Avellaneda-Stoikov Test Output
    run_backtest(
        file_path="src/results/as_test.csv",
        strategy_name="Avellaneda-Stoikov",
        trade_size=0.1
    )

    # 2. Evaluate XGBoost Test Output
    run_backtest(
        file_path="src/results/xgb_test_predictions.csv",
        strategy_name="XGBoost ML",
        trade_size=0.1
    )

if __name__ == "__main__":
    main()