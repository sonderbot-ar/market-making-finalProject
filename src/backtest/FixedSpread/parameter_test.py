import pandas as pd
import numpy as np
from src.strategies.fixed_spread_mm import FixedSpreadMM

def run_grid_search(file_path: str, nrows: int = 10000):
    df = pd.read_parquet(file_path).head(nrows)
    final_mid_price = df.iloc[-1]['mid_price']
    
    spread_candidates = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    trade_size = 0.001
    max_inventory_limit = 0.005
    fee_rate = 0.0002
    starting_cash = 100000.0
    
    summary_results = []
    
    print(f"{'Half-Spread ($)':<18}{'Total Fills':<15}{'Max Abs Inv (BTC)':<20}{'Final PnL ($)':<15}")
    print("-" * 68)
    
    for spread in spread_candidates:
        mm = FixedSpreadMM(
            half_spread=spread, 
            trade_size=trade_size,
            max_inventory=max_inventory_limit,
            fee_rate=fee_rate,
            starting_cash=starting_cash
        )

        portfolio_values = []
        abs_inventories = []
        
        for row in df.itertuples():
            mm.update_quotes(row.mid_price)
            mm.process_trade(row.trade_price)
            
            # Record state at each tick
            portfolio_values.append(mm.get_portfolio_value(row.mid_price))
            abs_inventories.append(abs(mm.inventory))
                    
        # 1. Peak-to-Trough Max Drawdown Calculation
        peak = starting_cash
        max_drawdown = 0.0
        for pv in portfolio_values:
            if pv > peak:
                peak = pv
            drawdown = peak - pv
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        total_fills = mm.bid_fills + mm.ask_fills
        fill_rate = total_fills / nrows # % of market ticks we participated in
        average_abs_inventory = np.mean(abs_inventories)
        max_abs_inventory = np.max(abs_inventories)
        
        final_portfolio_value = mm.get_portfolio_value(final_mid_price)
        final_net_pnl = mm.calculate_pnl(final_mid_price)
        ending_inventory = mm.inventory

        summary_results.append({
            'Half-Spread': spread,
            'final_net_pnl': final_net_pnl,
            'fees_paid': mm.fees_paid,
            'bid_fills': mm.bid_fills,
            'ask_fills': mm.ask_fills,
            'total_fills': total_fills,
            'ending_inventory': ending_inventory,
            'average_abs_inventory': average_abs_inventory,
            'max_abs_inventory': max_abs_inventory,
            'max_drawdown': max_drawdown,
            'fill_rate': fill_rate,
            'final_portfolio_value': final_portfolio_value
        })
    
    results_df = pd.DataFrame(summary_results)

    results_df.set_index('Half-Spread', inplace=True)
    print("\n--- STANDARDIZED METRICS REPORT ---")
    print(results_df.T.round(4).to_string())
    
    return results_df

if __name__ == "__main__":
    file_path = "/Users/andresrodartee/market-making-finalProject/data/sample/BTCUSDT_2024-03-27_merged.parquet"
    run_grid_search(file_path)