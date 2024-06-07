import time
import threading
import luno_python.client as luno
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import json
import logging
import signal
import sys
from scipy.stats import linregress

# Constants
API_KEY = 'xxx'
API_SECRET = 'xxx'
PAIR = 'XBTZAR'
AMOUNT = 0.0001  # Example amount of BTC to buy/sell
RANGE = 20
THRESHOLD = 0.1

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
            response = client.get_order_book_full(pair=PAIR)
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

# Function to calculate confidence based on order book history
def calculate_confidence(current_order_book, current_price):
    average_confidence = 0
    for i in range(1, RANGE):
        currency_range = i * 1000
        asks_within_range = [ask for ask in current_order_book['asks'] if current_price - currency_range <= float(ask['price']) <= current_price + currency_range]
        bids_within_range = [bid for bid in current_order_book['bids'] if current_price - currency_range <= float(bid['price']) <= current_price + currency_range]

        total_supply_ = sum(float(ask['volume']) for ask in asks_within_range)
        total_demand_ = sum(float(bid['volume']) for bid in bids_within_range)

        if total_supply_ + total_demand_ == 0:
            confidence_ = 0.5
        else:
            confidence_ = total_demand_ / (total_supply_ + total_demand_)
            slope_confidence_ = calculate_slope_confidence(asks_within_range, bids_within_range)
            confidence_ += slope_confidence_
            confidence_ = confidence_ / 2
        average_confidence += confidence_
    average_confidence = average_confidence / RANGE
    return average_confidence

# Function to calculate confidence based on slope of order book data
def calculate_slope_confidence(asks, bids):
    ask_prices = np.array([float(ask['price']) for ask in asks])
    ask_volumes = np.array([float(ask['volume']) for ask in asks])
    bid_prices = np.array([float(bid['price']) for bid in bids])
    bid_volumes = np.array([float(bid['volume']) for bid in bids])

    # Check if ask_prices or bid_prices are empty
    if len(ask_prices) == 0 or len(bid_prices) == 0:
        return 0.5  # or another default value

    cumulative_ask_volumes = np.cumsum(ask_volumes)
    cumulative_bid_volumes = np.cumsum(bid_volumes)

    ask_slope, _, _, _, _ = linregress(ask_prices, cumulative_ask_volumes)
    bid_slope, _, _, _, _ = linregress(bid_prices, cumulative_bid_volumes)

    if ask_slope + (-1 * bid_slope) == 0:
        slope_confidence = 0.5
    else:
        slope_confidence = (-1 * bid_slope) / (ask_slope + (-1 * bid_slope))
    return slope_confidence


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
        fee_percentage = 0.0 # float(fee_info['maker_fee'])
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
    old_confidence = None
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
        if not order_book:
            logging.error('Failed to retrieve order book')
            continue

        conf_delta = 0
        confidence = calculate_confidence(order_book, float(ticker_data['bid']))
        if old_confidence is not None:
            conf_delta = confidence - old_confidence 
        old_confidence = confidence
        logging.info(f" ")
        logging.info(f"BTC Price: R {float(ticker_data['bid'])}")
        logging.info(f'Confidence in BTC: {confidence}')
        logging.info(f'Confidence Delta: {conf_delta}')

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
