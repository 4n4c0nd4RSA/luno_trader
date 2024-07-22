import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
import time
from matplotlib.animation import FuncAnimation
from matplotlib.dates import DateFormatter
from config import PAIR, API_CALL_DELAY, PRICE_CONFIDENCE_THRESHOLD, MARKET_MOMENTUM_INDICATOR_THRESHOLD, MARKET_PERCEPTION_THRESHOLD, AVERAGE_WINDOW_SIZE
from luno_zar_trader import fetch_trade_history, process_data, calculate_price_slope, extract_base_currency
import tzlocal

VERSION = '1.0.2'

# Get the local timezone
local_tz = tzlocal.get_localzone()

# Global variables
price_line = None
trend_line = None
short_trend_line = None
market_momentum_indicator_line = None
price_confidence_line = None
market_perception_line = None
mmi_delta_line = None
market_momentum_indicator_text = None
mmi_delta_text = None

market_momentum_indicator_old = None

# Global variables to store data
market_momentum_indicator_data = []
price_confidence_data = []
market_perception_data = []
mmi_delta_data = []

# Global variables for axis limits
ax1_xlim = None
ax1_ylim = None
ax2_xlim = None
ax3_xlim = None
ax3_ylim = None

def update_plot(frame):
    global price_line, trend_line, short_trend_line, market_momentum_indicator_line, price_confidence_line, market_perception_line, mmi_delta_line
    global mmi_delta_text
    global market_momentum_indicator_data, price_confidence_data, market_perception_data, mmi_delta_data
    global ax1_xlim, ax1_ylim, ax2_xlim, ax3_xlim, ax3_ylim
    global market_momentum_indicator_old

    max_retries = 3
    retry_delay = 5

    for _ in range(max_retries):
        try:
            candles, short_candles, all_candles = fetch_trade_history(PAIR)
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
        price_line, = ax1.plot(df_all.index, df_all['Price'], label='Price')
    else:
        price_line.set_data(df_all.index, df_all['Price'])

    # Calculate average points for trend linesAVERAGE_WINDOW_SIZE
    start_avg_price = df['Price'].iloc[:AVERAGE_WINDOW_SIZE].mean()
    start_avg_time = df.index[:AVERAGE_WINDOW_SIZE].mean()
    end_avg_price = df['Price'].iloc[-AVERAGE_WINDOW_SIZE:].mean()
    end_avg_time = df.index[-AVERAGE_WINDOW_SIZE:].mean()

    short_start_avg_price = df_short['Price'].iloc[:AVERAGE_WINDOW_SIZE].mean()
    short_start_avg_time = df_short.index[:AVERAGE_WINDOW_SIZE].mean()
    short_end_avg_price = df_short['Price'].iloc[-AVERAGE_WINDOW_SIZE:].mean()
    short_end_avg_time = df_short.index[-AVERAGE_WINDOW_SIZE:].mean()

    # Update or create trend lines
    if trend_line is None:
        trend_line, = ax1.plot([start_avg_time, end_avg_time], [start_avg_price, end_avg_price], label='Market Perception', linestyle='--')
    else:
        trend_line.set_data([start_avg_time, end_avg_time], [start_avg_price, end_avg_price])

    if short_trend_line is None:
        short_trend_line, = ax1.plot([short_start_avg_time, short_end_avg_time], [short_start_avg_price, short_end_avg_price], label='Price Confidence', linestyle='--')
    else:
        short_trend_line.set_data([short_start_avg_time, short_end_avg_time], [short_start_avg_price, short_end_avg_price])

    mapped_value = calculate_price_slope(df)
    short_mapped_value = 1 - calculate_price_slope(df_short)
    market_momentum_indicator = (short_mapped_value + mapped_value) / 2
    
    if market_momentum_indicator_old is None:
        market_momentum_indicator_old = market_momentum_indicator
        market_momentum_indicator_delta = 0
    else:
        market_momentum_indicator_delta = market_momentum_indicator - market_momentum_indicator_old
    
    market_momentum_indicator_old = market_momentum_indicator

    # Record data
    current_time = df_all.index[-1]
    market_momentum_indicator_data.append((current_time, market_momentum_indicator))
    price_confidence_data.append((current_time, short_mapped_value))
    market_perception_data.append((current_time, mapped_value))
    mmi_delta_data.append((current_time, market_momentum_indicator_delta))

    # Keep only the last 100 data points to avoid cluttering
    market_momentum_indicator_data = market_momentum_indicator_data[-100:]
    price_confidence_data = price_confidence_data[-100:]
    market_perception_data = market_perception_data[-100:]
    mmi_delta_data = mmi_delta_data[-100:]

    # Update or create the lines on the second graph
    if price_confidence_line is None:
        price_confidence_line, = ax2.plot(*zip(*price_confidence_data), label=f'Price Confidence ({short_mapped_value:.2f})', color='#c76eff')
    else:
        price_confidence_line.set_data(*zip(*price_confidence_data))
        price_confidence_line.set_label(f'Price Confidence ({short_mapped_value:.2f})')

    if market_momentum_indicator_line is None:
        market_momentum_indicator_line, = ax2.plot(*zip(*market_momentum_indicator_data), label=f'Market Momentum Indicator ({market_momentum_indicator:.2f})', color='#f51bbe')
    else:
        market_momentum_indicator_line.set_data(*zip(*market_momentum_indicator_data))
        market_momentum_indicator_line.set_label(f'Market Momentum Indicator ({market_momentum_indicator:.2f})')

    if market_perception_line is None:
        market_perception_line, = ax2.plot(*zip(*market_perception_data), label=f'Market Perception ({mapped_value:.2f})', color='purple')
    else:
        market_perception_line.set_data(*zip(*market_perception_data))
        market_perception_line.set_label(f'Market Perception ({mapped_value:.2f})')

    # Update or create the MMI delta line on the third graph
    if mmi_delta_line is None:
        mmi_delta_line, = ax3.plot(*zip(*mmi_delta_data), label='MMI Delta', color='purple')
    else:
        mmi_delta_line.set_data(*zip(*mmi_delta_data))

    # Update or create text annotations
    if mmi_delta_text is None:
        mmi_delta_text = ax3.text(0.05, 0.90, '', transform=ax3.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    mmi_delta_text.set_text(f'MMI Delta: {market_momentum_indicator_delta:.5f}')

    # Update axis limits and labels
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.set_ylim(0, 1)  # Set fixed y-axis limits for ax2
    ax3.relim()
    ax3.autoscale_view()

    # Store the current axis limits if they haven't been set yet
    if ax1_xlim is None:
        ax1_xlim = ax1.get_xlim()
        ax1_ylim = ax1.get_ylim()
        ax2_xlim = ax2.get_xlim()
        ax3_xlim = ax3.get_xlim()
        ax3_ylim = ax3.get_ylim()

    ax1.set_title(f'{extract_base_currency(PAIR)}/ZAR Price History')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (ZAR)')
    ax1.grid(True)

    ax2.set_title('Market Indicators')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Indicator Value')
    ax2.grid(True)

    ax3.set_title('MMI Delta')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Delta Value')
    ax3.grid(True)

    # Update legend
    ax1.legend(loc='center left', bbox_to_anchor=(0, 0.5))
    ax2.legend(loc='center left', bbox_to_anchor=(0, 0.5))
    ax3.legend(loc='center left', bbox_to_anchor=(0, 0.5))

    # Format x-axis labels
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M', tz=local_tz))
        plt.sca(ax)
        plt.xticks(rotation=45)

def reset_zoom(event):
    global ax1_xlim, ax1_ylim, ax2_xlim, ax3_xlim, ax3_ylim
    ax1.set_xlim(ax1_xlim)
    ax1.set_ylim(ax1_ylim)
    ax2.set_xlim(ax2_xlim)
    ax2.set_ylim(0, 1)  # Reset y-axis limits for ax2 to fixed values
    ax3.set_xlim(ax3_xlim)
    ax3.set_ylim(ax3_ylim)
    plt.draw()

# Set up the plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 21), sharex=True)
fig.canvas.manager.set_window_title(f'Luno Price Confidence Graph - v{VERSION}')

ax2.set_ylim(0, 1)  # Set initial fixed y-axis limits for ax2
ax2.axhline(y=1, color='black', alpha=0.7)
ax2.axhline(y=0.5, color='gray', alpha=0.7)
ax2.axhline(y=0, color='black', alpha=0.7)

ax2.axhline(y=0.5 + PRICE_CONFIDENCE_THRESHOLD, color='g', linestyle='--', label=f'PC Buy Limit ({0.5 + PRICE_CONFIDENCE_THRESHOLD})')
ax2.axhline(y=0.5 + MARKET_MOMENTUM_INDICATOR_THRESHOLD, color='#0db542', linestyle='--', label=f'MMI Buy Limit ({0.5 + MARKET_MOMENTUM_INDICATOR_THRESHOLD})')
ax2.axhline(y=0.5 + MARKET_PERCEPTION_THRESHOLD, color='lime', linestyle='--', label=f'MP Buy Limit ({0.5 + MARKET_PERCEPTION_THRESHOLD})')
ax2.axhline(y=0.5, color='black', linestyle='--', label='Midpoint (0.5)')
ax2.axhline(y=0.5 - MARKET_PERCEPTION_THRESHOLD, color='pink', linestyle='--', label=f'MP Sell Limit ({0.5 - MARKET_PERCEPTION_THRESHOLD})')
ax2.axhline(y=0.5 - MARKET_MOMENTUM_INDICATOR_THRESHOLD, color='#f5b8cf', linestyle='--', label=f'MMI Sell Limit ({0.5 - MARKET_MOMENTUM_INDICATOR_THRESHOLD})')
ax2.axhline(y=0.5 - PRICE_CONFIDENCE_THRESHOLD, color='r', linestyle='--', label=f'PC Sell Limit ({0.5 - PRICE_CONFIDENCE_THRESHOLD})')

# Manually adjust the subplot layout
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95, hspace=0.3)

# Add custom home button with adjusted position
button_ax = fig.add_axes([0.85, 0.01, 0.1, 0.03])
home_button = Button(button_ax, 'Reset Zoom')
home_button.on_clicked(reset_zoom)

# Create the animation
ani = FuncAnimation(fig, update_plot, interval=API_CALL_DELAY*1000, cache_frame_data=False)

# Show the plot
plt.show()