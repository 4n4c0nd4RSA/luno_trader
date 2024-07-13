import matplotlib.pyplot as plt
import numpy as np
import luno_python.client as luno
import time
import logging
from matplotlib.animation import FuncAnimation
from luno_btc_zar_trader import RANGE

# Constants
PAIR = 'XBTZAR'
PRICE_DELTA_VALUE = 1000 * RANGE
UPDATE_INTERVAL = 5000  # Update every 5000 milliseconds (5 seconds)
VERSION = '1.0.0'

# Initialize logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s %(message)s')

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
    if ticker_data is None:
        logging.error('Failed to retrieve ticker data')
        return

    current_price = float(ticker_data['bid'])

    confidences = []
    labels = []

    for i in range(1, int(PRICE_DELTA_VALUE/1000)):
        PRICE_DELTA = i * 1000
        
        asks_within_range = [ask for ask in order_book['asks'] if current_price - PRICE_DELTA <= float(ask['price']) <= current_price + PRICE_DELTA]
        bids_within_range = [bid for bid in order_book['bids'] if current_price - PRICE_DELTA <= float(bid['price']) <= current_price + PRICE_DELTA]

        asks_ = asks_within_range
        bids_ = bids_within_range
        total_supply_ = sum(float(ask['volume']) for ask in asks_)
        total_demand_ = sum(float(bid['volume']) for bid in bids_)

        # Calculate confidence
        confidence_100 = total_demand_ / (total_supply_ + total_demand_)

        confidences.append(confidence_100)
        labels.append(i)

    # Calculate the average confidence
    average_confidence = np.mean(confidences)

    # Clear the current plot
    plt.cla()

    # Plotting
    plt.bar(labels, confidences, color='blue')
    plt.ylim(0, 1)
    plt.xlabel('Per R1000 Range')
    plt.ylabel('Confidence')
    plt.title('Confidence by Price Range')
    plt.grid(True)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Neutral Confidence')
    plt.axhline(y=average_confidence, color='green', linestyle='-', label=f'Average Confidence: {average_confidence:.2f}')
    plt.legend()

# Set up the plot
fig, ax = plt.subplots()
fig.canvas.manager.set_window_title(f'Luno Confidence Graph - v{VERSION}')

# Create the animation
ani = FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL)

# Show the plot
plt.show()
