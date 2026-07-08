import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_simulation_window(file_path: str, nrows: int = 10000):
    df = pd.read_parquet(file_path).head(nrows)
    
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    df['trendline'] = df['mid_price'].rolling(window=100).mean()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['datetime'], df['mid_price'], 
            label='Instantaneous Mid-Price', 
            color='#2c3e50', alpha=0.5, linewidth=1)
    
    ax.plot(df['datetime'], df['trendline'], 
            label='100-Tick Rolling Trend', 
            color='#e74c3c', linewidth=2)
    
    ax.set_title(
        'BTC-USD Microstructure: 10,000 Tick Simulation Window', 
        fontsize=14, fontweight='bold', pad=15
    )
    ax.set_xlabel('Time (UTC)', fontsize=12, fontweight='bold')
    ax.set_ylabel('BTC/USD Price ($)', fontsize=12, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    
    ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    
    output_filename = "simulation_trend_analysis.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    file_path = "/Users/andresrodartee/market-making-finalProject/data/sample/BTCUSDT_2024-03-27_merged.parquet"
    plot_simulation_window(file_path)