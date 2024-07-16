import matplotlib.pyplot as plt
import pandas as pd
import time
from matplotlib.animation import FuncAnimation
from matplotlib.dates import DateFormatter
from luno_btc_zar_trader import fetch_trade_history, process_data, calculate_price_slope
import tzlocal

VERSION = '1.0.1'

# Get the local timezone
local_tz = tzlocal.get_localzone()

# Global variables
price_line = None
trend_line = None
short_trend_line = None
price_confidence_text = None
short_price_confidence_text = None

def update_plot(frame):
    global price_line, trend_line, short_trend_line, price_confidence_text, short_price_confidence_text

    max_retries = 3
    retry_delay = 5

    for _ in range(max_retries):
        try:
            candles, short_candles, all_candles = fetch_trade_history()
            break
        except Exception as e:
            print(f"Error fetching trade history: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    else:
        print("Failed to fetch trade history after multiple attempts.")
        return

    df = process_data(candles)
    df_all = process_data(all_candles)
    df_short = process_data(short_candles)
    
    # Convert timestamps to pandas datetime
    df.index = pd.to_datetime(df.index).tz_convert(local_tz)
    df_all.index = pd.to_datetime(df_all.index).tz_convert(local_tz)
    df_short.index = pd.to_datetime(df_short.index).tz_convert(local_tz)
    
    # Update or create the price line

    if price_line is None:
        price_line, = ax.plot(df_all.index, df_all['Price'], label='Price')
    else:
        price_line.set_data(df_all.index, df_all['Price'])

    # Calculate and update trend lines
    start_price, end_price = df['Price'].iloc[0], df['Price'].iloc[-1]
    short_start_price, short_end_price = df_short['Price'].iloc[0], df_short['Price'].iloc[-1]

    if trend_line is None:
        trend_line, = ax.plot([df.index[0], df.index[-1]], [start_price, end_price], label='Market Perception', linestyle='--')
    else:
        trend_line.set_data([df.index[0], df.index[-1]], [start_price, end_price])

    if short_trend_line is None:
        short_trend_line, = ax.plot([df_short.index[0], df_short.index[-1]], [short_start_price, short_end_price], label='Price Confidence', linestyle='--')
    else:
        short_trend_line.set_data([df_short.index[0], df_short.index[-1]], [short_start_price, short_end_price])

    mapped_value = calculate_price_slope(df)
    short_mapped_value = 1 - calculate_price_slope(df_short)
    
    # Update or create text annotations
    if price_confidence_text is None:
        price_confidence_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    price_confidence_text.set_text(f'Market Perception: {mapped_value:.5f}')

    if short_price_confidence_text is None:
        short_price_confidence_text = ax.text(0.05, 0.90, '', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    short_price_confidence_text.set_text(f'Price Confidence: {short_mapped_value:.5f}')

    # Update axis limits and labels
    ax.relim()
    ax.autoscale_view()
    ax.set_title('BTC/ZAR Price History')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (ZAR)')
    ax.grid(True)

    # Update legend
    ax.legend(loc='center left', bbox_to_anchor=(0, 0.5))

    # Format x-axis labels
    plt.gcf().autofmt_xdate()  # Rotation
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M', tz=local_tz))

    # Adjust the layout
    plt.tight_layout()

# Set up the plot
fig, ax = plt.subplots(figsize=(14, 7))
fig.canvas.manager.set_window_title(f'Luno Price Confidence Graph - v{VERSION}')

# Create the animation
UPDATE_INTERVAL = 5000  # Update every 5000 milliseconds (5 seconds)
ani = FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL, cache_frame_data=False)

# Show the plot
plt.show()