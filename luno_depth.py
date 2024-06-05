import matplotlib.pyplot as plt
import numpy as np
import luno_python.client as luno
import time
import logging
from matplotlib.animation import FuncAnimation

# Constants
API_KEY = 'xxx'
API_SECRET = 'xxx'
PAIR = 'XBTZAR'
RANGE = 200
UPDATE_INTERVAL = 5000  # Update every 5000 milliseconds (5 seconds)

# Initialize logging
logging.basicConfig(filename='depth_script.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Initialize the Luno API client
client = luno.Client(api_key_id=API_KEY, api_key_secret=API_SECRET)

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

# Function to update the plot
def update_plot(frame):
    order_book = get_order_book()
    if order_book is None:
        logging.error('Failed to retrieve order book')
        return

    # Parse the order book to get asks and bids
    asks = order_book['asks'][:RANGE]
    bids = order_book['bids'][:RANGE]

    # Convert to numpy arrays for easy processing
    ask_prices = np.array([float(ask['price']) for ask in asks])
    ask_volumes = np.array([float(ask['volume']) for ask in asks])
    bid_prices = np.array([float(bid['price']) for bid in bids])
    bid_volumes = np.array([float(bid['volume']) for bid in bids])

    # Cumulative sums for depth
    cumulative_ask_volumes = np.cumsum(ask_volumes)
    cumulative_bid_volumes = np.cumsum(bid_volumes)

    # Clear the current plot
    plt.cla()

    # Plotting
    plt.plot(ask_prices, cumulative_ask_volumes, label='Asks', color='red')
    plt.plot(bid_prices, cumulative_bid_volumes, label='Bids', color='green')
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
