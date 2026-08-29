import numpy as np
import pandas as pd
import math

def run_delta_neutral_backtest(merged_parquet_path, output_csv_path):
    print("Loading master market state...")
    df = pd.read_parquet(merged_parquet_path)
    N = len(df)
    
    # --- Engine Parameters ---
    gamma = 0.1         # Risk aversion parameter
    kappa = 1.5         # Order flow decay parameter
    A = 1.0             # Order arrival intensity
    inventory_limit = 10 
    
    # --- Output Arrays ---
    out_Inventory = np.zeros(N)
    out_Cash = np.zeros(N)
    out_Wealth = np.zeros(N)
    out_Utility = np.zeros(N)
    out_PnL = np.zeros(N)
    out_ReservationPrice = np.zeros(N)
    out_OptimalSpread = np.zeros(N)
    out_InventorySkew = np.zeros(N)
    out_Bid = np.zeros(N)
    out_Ask = np.zeros(N)
    out_P_BidFill = np.zeros(N)
    out_P_AskFill = np.zeros(N)
    out_Sigma = np.zeros(N)
    out_Kappa = np.full(N, kappa)
    out_Trend = np.zeros(N)
    out_BuyEnabled = np.ones(N, dtype=bool)
    out_SellEnabled = np.ones(N, dtype=bool)
    out_InventoryLimit = np.full(N, inventory_limit)
    
    # --- State Tracking ---
    current_inventory = 0
    cash = 100000.0
    
    # Extract arrays (using suffixes from the merge script)
    spot_bids = df['best_bid_spot'].values
    spot_asks = df['best_ask_spot'].values
    futures_bids = df['best_bid_futures'].values
    futures_asks = df['best_ask_futures'].values
    mid_prices = df['mid_price'].values
    sigmas = df['rolling_volatility'].values
    trends = df['log_return'].values
    
    print("Executing high-frequency replay...")
    for i in range(N):
        mid = mid_prices[i]
        sigma = sigmas[i] if not np.isnan(sigmas[i]) and sigmas[i] > 0 else 0.0001
        
        # 1. Quoting Leg: Avellaneda-Stoikov calculations
        reservation_price = mid - (current_inventory * gamma * (sigma ** 2))
        optimal_spread = (2 / gamma) * math.log(1 + (gamma / kappa)) if gamma > 0 else 0.01
        
        bid_quote = reservation_price - (optimal_spread / 2)
        ask_quote = reservation_price + (optimal_spread / 2)
        
        # 2. Market Impact: Poisson Fill Probabilities
        delta_bid = max(0.0, mid - bid_quote)
        delta_ask = max(0.0, ask_quote - mid)
        
        p_bid_fill = min(1.0, A * math.exp(-kappa * delta_bid))
        p_ask_fill = min(1.0, A * math.exp(-kappa * delta_ask))
        
        # 3. Execution & Delta-Neutral Hedge

        hedge_threshold = 5
        
        bid_filled = (np.random.random() < p_bid_fill) and (current_inventory < inventory_limit)
        ask_filled = (np.random.random() < p_ask_fill) and (current_inventory > -inventory_limit)
        
        if bid_filled:
            current_inventory += 1
            cash -= bid_quote
            
        if ask_filled:
            current_inventory -= 1
            cash += ask_quote

        # Failsafe Taker Hedge: Only cross the Futures spread if threshold is breached
        if current_inventory >= hedge_threshold:
            # We are too long. Market Sell exactly the threshold amount on Futures.
            cash += (futures_bids[i] * current_inventory)
            current_inventory = 0
            
        elif current_inventory <= -hedge_threshold:
            # We are too short. Market Buy exactly the threshold amount on Futures.
            cash -= (futures_asks[i] * abs(current_inventory))
            current_inventory = 0
            
        # 4. Wealth & PnL
        wealth = cash + (current_inventory * mid)
        exponent = np.clip(-gamma * wealth, -700, 700)  # Prevent overflow in exp
        utility = -math.exp(exponent)
        
        # 5. Log Environment State
        out_Inventory[i] = current_inventory
        out_Cash[i] = cash
        out_Wealth[i] = wealth
        out_Utility[i] = utility
        out_PnL[i] = wealth - 100000.0
        out_ReservationPrice[i] = reservation_price
        out_OptimalSpread[i] = optimal_spread
        out_InventorySkew[i] = mid - reservation_price
        out_Bid[i] = bid_quote
        out_Ask[i] = ask_quote
        out_P_BidFill[i] = p_bid_fill
        out_P_AskFill[i] = p_ask_fill
        out_Sigma[i] = sigma
        out_Trend[i] = trends[i]
        out_BuyEnabled[i] = current_inventory < inventory_limit
        out_SellEnabled[i] = current_inventory > -inventory_limit

    print("Compiling Reinforcement Learning CSV...")
    results_df = pd.DataFrame({
        'Inventory': out_Inventory,
        'Cash': out_Cash,
        'Wealth': out_Wealth,
        'Utility': out_Utility,
        'PnL': out_PnL,
        'ReservationPrice': out_ReservationPrice,
        'OptimalSpread': out_OptimalSpread,
        'InventorySkew': out_InventorySkew,
        'Bid': out_Bid,
        'Ask': out_Ask,
        'P_BidFill': out_P_BidFill,
        'P_AskFill': out_P_AskFill,
        'Sigma': out_Sigma,
        'Kappa': out_Kappa,
        'Trend': out_Trend,
        'BuyEnabled': out_BuyEnabled,
        'SellEnabled': out_SellEnabled,
        'InventoryLimit': out_InventoryLimit
    })
    
    results_df.to_csv(output_csv_path, index=False)
    print(f"Successfully exported state space to {output_csv_path}")

# --- TRIGGER LINE ---
# Ensure you uncomment the line below and that the filenames match your local directory
run_delta_neutral_backtest("/Users/andresrodartee/market-making-finalProject/data/raw/btcusdt_master_merged.parquet", "delta_neutral_v1.csv")