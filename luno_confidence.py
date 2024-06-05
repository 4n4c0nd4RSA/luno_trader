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
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s %(message)s')

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
    asks = order_book['asks']
    bids = order_book['bids']

    confidences = []
    labels = []

    for i in range(1, RANGE):
        asks_ = asks[:i]
        bids_ = bids[:i]
        total_supply_ = sum(float(ask['volume']) for ask in asks_)
        total_demand_ = sum(float(bid['volume']) for bid in bids_)

        # Calculate confidence
        confidence_100 = total_demand_ / (total_supply_ + total_demand_)

        confidences.append(confidence_100)
        labels.append(i)

    # Clear the current plot
    plt.cla()

    # Plotting
    plt.bar(labels, confidences, color='blue')
    plt.ylim(0, 1)
    plt.xlabel('Order Book')
    plt.ylabel('Confidence')
    plt.title('Confidence: First 100 Orders to Full Book')
    plt.grid(True)

# Set up the plot
fig, ax = plt.subplots()

# Create the animation
ani = FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL)

# Show the plot
plt.show()
