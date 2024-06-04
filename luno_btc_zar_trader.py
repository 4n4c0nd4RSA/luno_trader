import time
import threading
import luno_python.client as luno
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import logging
import signal
import sys

# Constants
API_KEY = 'xxx'
API_SECRET = 'xxx'
PAIR = 'XBTZAR'
AMOUNT = 0.0003  # Example amount of BTC to buy/sell

# Initialize logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Initialize the Luno API client
client = luno.Client(api_key_id=API_KEY, api_key_secret=API_SECRET)

# Initialize wallet balances
ZAR_balance = 2000.0  # Initial ZAR balance
BTC_balance = 0.0     # Initial BTC balance

# Lists to store wallet values over time
time_steps = []
wallet_values = []
btc_values_in_zar = []
zar_values = []

# Function to get the order book
def get_order_book():
    retries = 5
    for i in range(retries):
        try:
            response = client.get_order_book(pair=PAIR)
            return response
        except Exception as e:
            logging.error(f'Error getting order book: {e}')
            time.sleep(2 ** i)  # Exponential backoff
    return None

# Function to read order book history from a file
def read_order_book_history():
    filename = 'order_book_history.json'
    try:
        with open(filename, 'r') as f:
            history = [json.loads(line) for line in f]
        return history
    except FileNotFoundError:
        logging.warning(f'History file {filename} not found.')
        return []
    except Exception as e:
        logging.error(f'Error reading history file: {e}')
        return []

# Function to save the order book to a file
def save_order_book_to_file(order_book):
    filename = 'order_book_history.json'
    current_time = time.time()
    with open(filename, 'a') as f:
        record = {
            'timestamp': current_time,
            'order_book': order_book
        }
        f.write(json.dumps(record) + '\n')

# Function to calculate confidence based on order book history
def calculate_confidence(current_order_book, history):
    if not history:
        return 0.5  # Neutral confidence if no history is available

    current_asks = {float(order['price']): float(order['volume']) for order in current_order_book['asks'][:100]}
    current_bids = {float(order['price']): float(order['volume']) for order in current_order_book['bids'][:100]}

    recent_history = history[-100:]  # Use the most recent 100 entries for comparison

    ask_confidence = 0
    bid_confidence = 0
    total_entries = len(recent_history)

    for entry in recent_history:
        historical_asks = {float(order['price']): float(order['volume']) for order in entry['order_book']['asks'][:100]}
        historical_bids = {float(order['price']): float(order['volume']) for order in entry['order_book']['bids'][:100]}

        # Compare historical asks with current asks
        for price, volume in historical_asks.items():
            if price in current_asks:
                ask_confidence += min(volume, current_asks[price]) / max(volume, current_asks[price])

        # Compare historical bids with current bids
        for price, volume in historical_bids.items():
            if price in current_bids:
                bid_confidence += min(volume, current_bids[price]) / max(volume, current_bids[price])

    # Normalize confidence values to be between 0 and 1
    ask_confidence /= total_entries
    bid_confidence /= total_entries

    # Combine ask and bid confidence to get overall confidence
    overall_confidence = (ask_confidence + bid_confidence) / 2
    return overall_confidence / 100.0

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

# Function to get fee information
def get_fee_info():
    retries = 5
    for i in range(retries):
        try:
            response = client.get_fee_info(pair=PAIR)
            return response
        except Exception as e:
            logging.error(f'Error getting fee information: {e}')
            time.sleep(2 ** i)  # Exponential backoff
    return None

# Function to determine the action (Buy, Sell, or Nothing) based on confidence
def determine_action(ticker_data, confidence):
    global ZAR_balance, BTC_balance

    # Calculate current BTC value in ZAR
    btc_to_zar = BTC_balance * float(ticker_data['bid'])
    total_value_zar = ZAR_balance + btc_to_zar

    # Calculate current BTC percentage of total value
    current_btc_percentage = btc_to_zar / total_value_zar
    logging.info(f'BTC %: {current_btc_percentage}%')

    # Define a threshold to avoid oscillation
    THRESHOLD = AMOUNT * float(ticker_data['bid']) / total_value_zar

    # Determine action based on the target confidence and threshold
    if current_btc_percentage < confidence - THRESHOLD:
        return 'Buy'
    elif current_btc_percentage > confidence + THRESHOLD:
        return 'Sell'
    else:
        return 'Nothing'

# Mock trade function to print what would happen in a trade
def mock_trade(order_type, amount, ticker_data, fee_info):
    global ZAR_balance, BTC_balance
    amount = float(amount)
    if order_type == 'Buy':
        price = float(ticker_data['ask'])
        fee_percentage = float(fee_info['maker_fee'])
        cost = price * amount
        fee = cost * fee_percentage
        total_cost = cost + fee
        if ZAR_balance >= total_cost:
            ZAR_balance -= total_cost
            BTC_balance += amount
            logging.info(f'Bought {amount} BTC at {price} ZAR/BTC')
            logging.info(f'Cost: {cost} ZAR')
            logging.info(f'Fee: {fee} ZAR')
            logging.info(f'Total Cost: {total_cost} ZAR')
        else:
            logging.warning('Insufficient ZAR balance to complete the buy order')
    elif order_type == 'Sell':
        price = float(ticker_data['bid'])
        fee_percentage = float(fee_info['taker_fee'])
        revenue = price * amount
        fee = revenue * fee_percentage
        total_revenue = revenue - fee
        if BTC_balance >= amount:
            BTC_balance -= amount
            ZAR_balance += total_revenue
            logging.info(f'Sold {amount} BTC at {price} ZAR/BTC')
            logging.info(f'Revenue: {revenue} ZAR')
            logging.info(f'Fee: {fee} ZAR')
            logging.info(f'Total Revenue: {total_revenue} ZAR')
        else:
            logging.warning('Insufficient BTC balance to complete the sell order')
    logging.info(f'New ZAR balance: {ZAR_balance}')
    logging.info(f'New BTC balance: {BTC_balance} ({BTC_balance * float(ticker_data["bid"])})')

# Function to update wallet values for plotting
def update_wallet_values(ticker_data):
    global ZAR_balance, BTC_balance, time_steps, wallet_values, btc_values_in_zar, zar_values

    # Calculate current BTC value in ZAR
    btc_to_zar = BTC_balance * float(ticker_data['bid'])
    total_value_zar = ZAR_balance + btc_to_zar

    # Store the current time and wallet value in ZAR
    current_time = time.time()
    time_steps.append(current_time)
    wallet_values.append(total_value_zar)
    btc_values_in_zar.append(btc_to_zar)
    zar_values.append(ZAR_balance)

# Function to plot wallet values over time
def plot_wallet_values():
    plt.plot(time_steps, wallet_values, label='Wallet Value in ZAR')
    plt.xlabel('Time')
    plt.ylabel('Wallet Value (ZAR)')
    plt.title('Wallet Value Over Time')
    plt.legend()
    plt.show()

# Function to update the plot
def update_plot(frame):
    plt.cla()  # Clear the current axes
    plt.plot(time_steps, wallet_values, label='Total Wallet Value in ZAR')
    plt.plot(time_steps, btc_values_in_zar, label='BTC Value in ZAR')
    plt.plot(time_steps, zar_values, label='ZAR Value')
    plt.xlabel('Time')
    plt.ylabel('Value (ZAR)')
    plt.title('Wallet Values Over Time')
    plt.legend()

# Graceful shutdown handler
def signal_handler(sig, frame):
    logging.info('Gracefully shutting down...')
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Main function to check the ticker and place orders
def trading_loop():
    global ZAR_balance, BTC_balance  # Ensure the main function knows about the global variables

    while True:
        fee_info = get_fee_info()
        if not fee_info:
            logging.error('Failed to retrieve fee information')
            continue

        ticker_data = get_ticker()
        if not ticker_data:
            logging.error('Failed to retrieve ticker data')
            continue

        order_book = get_order_book()
        if order_book:
            save_order_book_to_file(order_book)
        else:
            logging.error('Failed to retrieve order book')

        history = read_order_book_history()
        confidence = calculate_confidence(order_book, history)
        logging.info(f'Confidence in BTC: {confidence}')

        action = determine_action(ticker_data, confidence)

        if action == 'Buy':
            mock_trade('Buy', AMOUNT, ticker_data, fee_info)
        elif action == 'Sell':
            mock_trade('Sell', AMOUNT, ticker_data, fee_info)
        else:
            logging.info('No action taken')
        logging.info(f'Current Total balance: {ZAR_balance + BTC_balance * float(ticker_data["bid"])}')

        # Update wallet values regardless of the action
        update_wallet_values(ticker_data)

        time.sleep(5)

# Start the trading loop in a separate thread
trading_thread = threading.Thread(target=trading_loop)
trading_thread.daemon = True
trading_thread.start()

if __name__ == '__main__':
    fig = plt.figure()
    ani = FuncAnimation(fig, update_plot, interval=5000)  # Update every second
    plt.show()  # Show the plot
