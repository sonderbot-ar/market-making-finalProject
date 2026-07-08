import pandas as pd
import numpy as np
from src.strategies.fixed_spread_mm import FixedSpreadMM

def run_grid_search(file_path: str, nrows: int = 10000):
    df = pd.read_parquet(file_path).head(nrows)
    final_mid_price = df.iloc[-1]['mid_price']
    
    spread_candidates = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    trade_size = 0.01
    
    summary_results = []
    
    print(f"{'Half-Spread ($)':<18}{'Total Fills':<15}{'Max Abs Inv (BTC)':<20}{'Final PnL ($)':<15}")
    print("-" * 68)
    
    for spread in spread_candidates:
        mm = FixedSpreadMM(half_spread=spread, trade_size=trade_size)
        
        fills = 0
        max_inv = 0.0
        
        for row in df.itertuples():
            mm.update_quotes(row.mid_price)
            
            # Track state changes to count fills
            prev_inv = mm.inventory
            mm.process_trade(row.trade_price)
            
            if mm.inventory != prev_inv:
                fills += 1
                
            max_inv = max(max_inv, abs(mm.inventory))
            
        final_pnl = mm.calculate_pnl(final_mid_price)
        
        summary_results.append({
            'half_spread': spread,
            'total_fills': fills,
            'max_inventory': max_inv,
            'final_pnl': final_pnl
        })
        
        print(f"{spread:<18.2f}{fills:<15}{max_inv:<20.4f}${final_pnl:<14.4f}")
        
    return pd.DataFrame(summary_results)

if __name__ == "__main__":
    file_path = "/Users/andresrodartee/market-making-finalProject/data/sample/BTCUSDT_2024-03-27_merged.parquet"
    run_grid_search(file_path)