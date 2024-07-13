import luno_python.client
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from matplotlib.animation import FuncAnimation
from luno_btc_zar_trader import fetch_trade_history, process_data, calculate_price_slope

# Initialize the Luno API client
all_trades = []
VERSION = '1.0.0'

# Update the plot
def update_plot(frame):
    candles = None
    while candles == None:
        try:
            candles, short_candles = fetch_trade_history()
        except:
            time.sleep(5)
    df = process_data(candles)
    df_short = process_data(short_candles)
    
    ax.clear()
    ax.plot(df.index, df['Price'], label='Price')

    # Calculate the average angle straight line
    start_price = df['Price'].iloc[0]
    end_price = df['Price'].iloc[-1]
    short_start_price = df_short['Price'].iloc[0]
    short_end_price = df_short['Price'].iloc[-1]
    ax.plot([df.index[0], df.index[-1]], [start_price, end_price], label='Trend Line', linestyle='--')
    ax.plot([df_short.index[0], df_short.index[-1]], [short_start_price, short_end_price], label='Short Trend Line', linestyle='--')
    mapped_value = calculate_price_slope(df)
    short_mapped_value = 1-calculate_price_slope(df_short)
    
    # Add text annotation for the mapped value
    ax.text(0.05, 0.95, f'Price Confidence: {mapped_value:.5f}', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    ax.text(0.05, 0.90, f'Short Price Confidence: {short_mapped_value:.5f}', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    
    ax.set_title('BTC/ZAR Price History')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (ZAR)')
    ax.grid()

    # Create a legend that gravitates to the left
    ax.legend(loc='center left', bbox_to_anchor=(0, 0.5))

    # Adjust the layout
    plt.tight_layout()

# Set up the plot
fig, ax = plt.subplots(figsize=(14, 7))
fig.canvas.manager.set_window_title(f'Luno Price Confidence Graph - v{VERSION}')

# Create the animation
UPDATE_INTERVAL = 5000  # Update every 5000 milliseconds (5 seconds)
ani = FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL)

# Show the plot
plt.show()