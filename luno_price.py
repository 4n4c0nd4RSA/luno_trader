import luno_python.client
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from matplotlib.animation import FuncAnimation

# Initialize the Luno API client
client = luno_python.client.Client()
since = int(time.time()*1000)-23*60*60*1000
age_limit = int(time.time()*1000)-23*60*60*1000
all_trades = []
# Fetch BTC/ZAR price history
def fetch_trade_history(pair='XBTZAR'):
    global since, all_trades
    res = client.list_trades(pair=pair, since=since)
    all_trades.extend(res['trades'])
    while len(res['trades']) == 100:
        max_timestamp = max(entry['timestamp'] for entry in res['trades'])
        since = max_timestamp + 1
        res = client.list_trades(pair=pair, since=since)
        all_trades.extend(res['trades'])
    recent_trades = [trade for trade in all_trades if trade['timestamp'] >= age_limit]
    return recent_trades

# Process data into a DataFrame
def process_data(candles):
    data = []
    for candle in candles:
        timestamp = pd.to_datetime(candle['timestamp'], unit='ms')
        price = float(candle['price'])
        volume = float(candle['volume'])
        data.append([timestamp, price, volume])
    df = pd.DataFrame(data, columns=['Timestamp', 'Price', 'Volume'])
    return df.set_index('Timestamp')

# Update the plot
def update_plot(frame):
    candles = fetch_trade_history()
    df = process_data(candles)
    
    ax.clear()
    ax.plot(df.index, df['Price'], label='Price')

    # Calculate the average angle straight line
    start_price = df['Price'].iloc[0]
    end_price = df['Price'].iloc[-1]
    ax.plot([df.index[0], df.index[-1]], [start_price, end_price], label='Trend Line', linestyle='--')
    x_diff = (df.index[-1] - df.index[0]).total_seconds()  # Difference in seconds
    y_diff = end_price - start_price

    # Calculate the slope
    slope = y_diff / x_diff
    mapped_value = 1 / (1 + np.exp(-slope))
    
    # Add text annotation for the mapped value
    ax.text(0.05, 0.95, f'Price Confidence: {mapped_value:.5f}', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    
    ax.set_title('BTC/ZAR Price History')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (ZAR)')
    ax.legend()
    ax.grid()

# Set up the plot
fig, ax = plt.subplots(figsize=(14, 7))

# Create the animation
UPDATE_INTERVAL = 5000  # Update every 5000 milliseconds (5 seconds)
ani = FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL)

# Show the plot
plt.show()
