import numpy as np
import pandas as pd
from numba import njit

@njit
def fast_as_loop(bids, asks, vols, gamma, kappa, fee_rate, starting_cash, starting_inv):
    """
    Highly optimized tick-by-tick simulation of the Avellaneda-Stoikov model.
    """
    n = len(bids)
    pnl_series = np.zeros(n)
    inv_series = np.zeros(n)
    
    cash = starting_cash
    inventory = starting_inv
    
    total_fills = 0
    bid_fills = 0
    ask_fills = 0
    fees_paid = 0.0
    total_quotes = 0
    
    for i in range(n):
        market_bid = bids[i]
        market_ask = asks[i]
        sigma = vols[i]
        
        # If data is missing or volatility is zero, carry forward state
        if np.isnan(market_bid) or np.isnan(market_ask) or sigma <= 0:
            pnl_series[i] = cash + (inventory * market_bid) - starting_cash
            inv_series[i] = inventory
            continue
            
        mid_price = (market_bid + market_ask) / 2.0
        
        # Avellaneda-Stoikov Math
        reservation_price = mid_price - (inventory * gamma * (sigma ** 2))
        spread = (2.0 / gamma) * np.log(1.0 + (gamma / kappa))
        
        agent_bid = reservation_price - (spread / 2.0)
        agent_ask = reservation_price + (spread / 2.0)
        total_quotes += 2
        
        # Evaluate Fills (Assuming instantaneous execution if market crosses quotes)
        if market_bid <= agent_bid:
            inventory += 1
            fee = market_bid * fee_rate
            cash -= (market_bid + fee)
            fees_paid += fee
            bid_fills += 1
            total_fills += 1
            
        if market_ask >= agent_ask:
            inventory -= 1
            fee = market_ask * fee_rate
            cash += (market_ask - fee)
            fees_paid += fee
            ask_fills += 1
            total_fills += 1
            
        # Mark-to-market PnL based on current market bid
        pnl_series[i] = (cash + (inventory * market_bid)) - starting_cash
        inv_series[i] = inventory
        
    fills_data = {
        'total_fills': total_fills,
        'bid_fills': bid_fills,
        'ask_fills': ask_fills,
        'fees_paid': fees_paid,
        'total_quotes': total_quotes
    }
    
    return pnl_series, inv_series, fills_data

def simulate_baseline_as(df, params, fee_rate, starting_cash, starting_inv):
    """
    Wrapper function to plug seamlessly into your WFO Harness.
    """
    # Extract arrays
    bids = df['best_bid_spot'].to_numpy()
    asks = df['best_ask_spot'].to_numpy()
    
    # Assumes you have a 'rolling_volatility' column in your dataframe
    vols = df['rolling_volatility'].to_numpy() 
    
    # Extract params from grid dictionary
    gamma = params.get('gamma', 0.1)
    kappa = params.get('kappa', 1.5)
    
    # Execute Numba loop
    pnl, inv, fills = fast_as_loop(
        bids, asks, vols, gamma, kappa, fee_rate, starting_cash, starting_inv
    )
    
    return pnl, inv, fills