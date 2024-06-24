import matplotlib.pyplot as plt
import numpy as np
import luno_python.client as luno
import time
import logging
from matplotlib.animation import FuncAnimation
from scipy.stats import linregress

# Constants
PAIR = 'XBTZAR'
PRICE_DELTA = 400000
UPDATE_INTERVAL = 5000  # Update every 5000 milliseconds (5 seconds)

# Initialize logging
logging.basicConfig(filename='depth_script.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Initialize the Luno API client
client = luno.Client()

# Function to get the order book
def get_order_book():
    retries = 5
    for i in range(retries):
        try:
            response = client.get_order_book_full(pair=PAIR)
            return response
        except Exception as e:
            logging.error(f'Error getting order book: {e}')
            time.sleep(2 ** i)  # Exponential backoff
    return None

# Function to get the latest ticker information
def get_ticker():
    retries = 5
    for i in range(retries):
        try:
            response = client.get_ticker(pair=PAIR)
            return response
        except Exception as e:
            logging.error(f'Error getting ticker information: {e}')
            time.sleep(2 ** i)  # Exponential backoff
    return None

# Function to update the plot
def update_plot(frame):
    order_book = get_order_book()
    if order_book is None:
        logging.error('Failed to retrieve order book')
        return

    ticker_data = get_ticker()
    if (ticker_data is None):
        logging.error('Failed to retrieve ticker data')
        return

    current_price = float(ticker_data['bid'])
    
    # Filter asks and bids within the price range
    asks_within_range = [ask for ask in order_book['asks'] if current_price - PRICE_DELTA <= float(ask['price']) <= current_price + PRICE_DELTA]
    bids_within_range = [bid for bid in order_book['bids'] if current_price - PRICE_DELTA <= float(bid['price']) <= current_price + PRICE_DELTA]

    asks = asks_within_range
    bids = bids_within_range

    # Convert to numpy arrays for easy processing
    ask_prices = np.array([float(ask['price']) for ask in asks])
    ask_volumes = np.array([float(ask['volume']) for ask in asks])
    bid_prices = np.array([float(bid['price']) for bid in bids])
    bid_volumes = np.array([float(bid['volume']) for bid in bids])

    # Cumulative sums for depth
    cumulative_ask_volumes = np.cumsum(ask_volumes)
    cumulative_bid_volumes = np.cumsum(bid_volumes)

    # Linear regression for asks and bids
    ask_slope, ask_intercept, _, _, _ = linregress(ask_prices, cumulative_ask_volumes)
    bid_slope, bid_intercept, _, _, _ = linregress(bid_prices, cumulative_bid_volumes)

    # Clear the current plot
    plt.cla()

    # Plotting
    plt.plot(ask_prices, cumulative_ask_volumes, label='Asks', color='red')
    plt.plot(bid_prices, cumulative_bid_volumes, label='Bids', color='green')

    # Plotting the average angle lines
    plt.plot(ask_prices, ask_slope * ask_prices + ask_intercept, label='Asks Trendline', color='orange', linestyle='--')
    plt.plot(bid_prices, bid_slope * bid_prices + bid_intercept, label='Bids Trendline', color='blue', linestyle='--')

    # Calculate intercepts at the current price
    asks_intercept_at_current_price = ask_slope * current_price + ask_intercept
    bids_intercept_at_current_price = bid_slope * current_price + bid_intercept

    # Plot intercepts at the current price
    plt.scatter([current_price], [asks_intercept_at_current_price], color='orange', marker='o', label='Ask Intercept at Current Price')
    plt.scatter([current_price], [bids_intercept_at_current_price], color='blue', marker='o', label='Bid Intercept at Current Price')

    # Adding trendline values as text
    plt.text(0.05, 0.05, f'Asks Trendline: {ask_slope:.5f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', color='orange')
    plt.text(0.05, 0.10, f'Bids Trendline: {bid_slope:.5f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', color='blue')
    plt.text(0.05, 0.15, f'Asks Intercept at Current Price: {asks_intercept_at_current_price:.5f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', color='orange')
    plt.text(0.05, 0.20, f'Bids Intercept at Current Price: {bids_intercept_at_current_price:.5f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', color='blue')

    intercept_confidence = 1 / (1 + np.exp(-(0.125 * (bids_intercept_at_current_price - asks_intercept_at_current_price))))
    trendline_confidence = (-1*bid_slope)/((-1*bid_slope)+ask_slope)
    plt.text(0.05, 0.25, f'Trendline Confidence = {trendline_confidence:.5f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', color='green')
    plt.text(0.05, 0.30, f'Intercept Confidence = {intercept_confidence:.5f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', color='green')


    plt.xlabel('Price (ZAR)')
    plt.ylabel('Cumulative Volume (BTC)')
    plt.title('Order Book Depth')
    plt.legend()
    plt.grid(True)

# Set up the plot
fig, ax = plt.subplots()

# Create the animation
ani = FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL)

# Show the plot
plt.show()