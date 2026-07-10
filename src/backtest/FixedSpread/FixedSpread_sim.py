import pandas as pd
from src.strategies.fixed_spread_mm import FixedSpreadMM

def run_simulation(file_path: str, nrows: int = 10000):
    df = pd.read_parquet(file_path).head(nrows)

    mm = FixedSpreadMM(
        half_spread=2.0, 
        trade_size=0.001,
        max_inventory=0.005,
        fee_rate=0.0002,
        starting_cash=100000.0
    )

    results = []

    for row in df.itertuples():
        mm.update_quotes(row.mid_price)
        mm.process_trade(row.trade_price)

        results.append({
            'timestamp': row.timestamp,
            'mid_price': row.mid_price,
            'inventory': mm.inventory,
            'cash': mm.cash,
            'portfolio_value': mm.get_portfolio_value(row.mid_price),
            'total_pnl': mm.calculate_pnl(row.mid_price)
        })
    
    results_df = pd.DataFrame(results)

    print(results_df.tail())
    print(f'\nFinal Inventory Exposure: {mm.inventory:.4f} BTC')
    print(f"Final Cash: {mm.cash:.4f} USDT")
    print(f"Final Portfolio Value: {mm.get_portfolio_value(df.iloc[-1]['mid_price']):.4f} USDT")
    print(f"\nFinal Total PnL: ${mm.calculate_pnl(df.iloc[-1]['mid_price']):.4f}")

    return results_df

if __name__ == '__main__':
    file_path = "/Users/andresrodartee/market-making-finalProject/data/sample/BTCUSDT_2024-03-27_merged.parquet"
    run_simulation(file_path)