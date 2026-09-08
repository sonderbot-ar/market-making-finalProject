import numpy as np
import math
from numba import njit

@njit(nopython=True)
def fast_vpin_hedge_loop(spot_bids, spot_asks, futures_bids, futures_asks, 
                         mids, sigmas, vpins, gamma, kappa, A, 
                         hedge_threshold, vpin_threshold, fee_rate, 
                         starting_cash, starting_inv):
    """
    Numba-optimized tick-by-tick simulation of the VPIN Delta-Hedge model.
    """
    n = len(spot_bids)
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
        mid = mids[i]
        sigma = sigmas[i] if not np.isnan(sigmas[i]) and sigmas[i] > 0 else 0.0001
        vpin = vpins[i]
        
        # 1. Quoting Leg: Avellaneda-Stoikov calculations
        reservation_price = mid - (inventory * gamma * (sigma ** 2))
        optimal_spread = (2.0 / gamma) * math.log(1.0 + (gamma / kappa)) if gamma > 0 else 0.01
        
        bid_quote = reservation_price - (optimal_spread / 2.0)
        ask_quote = reservation_price + (optimal_spread / 2.0)
        total_quotes += 2
        
        # 2. Market Impact: Poisson Fill Probabilities
        delta_bid = max(0.0, mid - bid_quote)
        delta_ask = max(0.0, ask_quote - mid)
        
        p_bid_fill = min(1.0, A * math.exp(-kappa * delta_bid))
        p_ask_fill = min(1.0, A * math.exp(-kappa * delta_ask))

        maker_fee_rate = 0.0000  # 0 bps for providing liquidity
        taker_fee_rate = fee_rate # 2 bps for crossing the spread to hedge
        inventory_limit = hedge_threshold # Stop quoting if we reach the hedge threshold

        # 3. Probabilistic Execution[cite: 3]
        bid_filled = (np.random.random() < p_bid_fill) and (inventory < inventory_limit)
        ask_filled = (np.random.random() < p_ask_fill) and (inventory > -inventory_limit)
        
        if bid_filled:
            inventory += 1
            fee = bid_quote * maker_fee_rate  
            cash -= (bid_quote + fee)
            fees_paid += fee
            bid_fills += 1
            total_fills += 1
            
        if ask_filled:
            inventory -= 1
            fee = ask_quote * maker_fee_rate  
            cash += (ask_quote - fee)
            fees_paid += fee
            ask_fills += 1
            total_fills += 1

        # 4. Failsafe Taker Hedge[cite: 3]
        # Upgraded to trigger on standard inventory OR toxic order flow (VPIN)
        if inventory >= hedge_threshold or (inventory > 0 and vpin > vpin_threshold):
            fee = futures_bids[i] * inventory * taker_fee_rate
            cash += (futures_bids[i] * inventory) - fee
            fees_paid += fee
            inventory = 0
            
        elif inventory <= -hedge_threshold or (inventory < 0 and vpin > vpin_threshold):
            fee = futures_asks[i] * abs(inventory) * taker_fee_rate
            cash -= (futures_asks[i] * abs(inventory)) + fee
            fees_paid += fee
            inventory = 0
            
        # 5. Track State
        pnl_series[i] = (cash + (inventory * mid)) - starting_cash
        inv_series[i] = inventory
        
    
    return pnl_series, inv_series, total_fills, bid_fills, ask_fills, fees_paid, total_quotes

def simulate_vpin_hedge(df, params, fee_rate, starting_cash, starting_inv):
    """
    Wrapper function to plug this model seamlessly into the WFO Harness.
    """

    # df = df.iloc[:1000].copy()  # For testing purposes, limit to first 1000 rows. Remove in production.

    num_cols = [
        'best_bid_spot', 'best_ask_spot', 
        'best_bid_futures', 'best_ask_futures', 
        'rolling_volatility', 'vpin'
    ]
    
    # Backfill first, then fill remaining NaNs with 0.0
    df[num_cols] = df[num_cols].bfill().fillna(0.0)
    
    # Extract arrays to feed into the Numba loop
    spot_bids = df['best_bid_spot'].to_numpy()

    # Extract arrays to feed into the Numba loop
    spot_bids = df['best_bid_spot'].to_numpy(dtype=float)
    spot_asks = df['best_ask_spot'].to_numpy(dtype=float)
    futures_bids = df['best_bid_futures'].to_numpy(dtype=float)
    futures_asks = df['best_ask_futures'].to_numpy(dtype=float)
    mids = ((spot_bids + spot_asks) / 2.0)
    sigmas = df['rolling_volatility'].to_numpy(dtype=float)
    vpins = df['vpin'].to_numpy(dtype=float)
    
    # Extract parameters from the WFO grid search
    gamma = float(params.get('gamma', 0.1))
    kappa = float(params.get('kappa', 1.5))
    A = float(params.get('A', 1.0))
    hedge_threshold = float(params.get('hedge_threshold', 5.0))
    vpin_threshold = float(params.get('vpin_threshold', 0.8))
    

    pnl_series, inv_series, total_fills, bid_fills, ask_fills, fees_paid, total_quotes = fast_vpin_hedge_loop(
        spot_bids, spot_asks, futures_bids, futures_asks, mids, sigmas, vpins, 
        gamma, kappa, A, hedge_threshold, vpin_threshold, fee_rate, starting_cash, starting_inv
    )
    
    # Rebuild the dictionary in standard Python
    fills_data = {
        'total_fills': total_fills,
        'bid_fills': bid_fills,
        'ask_fills': ask_fills,
        'fees_paid': fees_paid,
        'total_quotes': total_quotes
    }
    
    return pnl_series, inv_series, fills_data
