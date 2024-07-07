import time
import os
import logging
import signal
import sys
import threading
import argparse
import luno_python.client as luno
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter
from scipy.stats import linregress 

# Constants
API_KEY = os.getenv('LUNO_API_KEY_ID')
API_SECRET = os.getenv('LUNO_API_KEY_SECRET')
PAIR = 'XBTZAR'
RANGE = 200
THRESHOLD = 0.1

# start time
start_time = time.gmtime()

# Initialize logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Initialize the Luno API client
client = luno.Client(api_key_id=API_KEY, api_key_secret=API_SECRET)

# Initialize wallet balances
ZAR_balance = 2000  # Initial ZAR balance
BTC_balance = 0  # Initial BTC balance

# Lists to store wallet values over time
time_steps = []
wallet_values = []
btc_values_in_zar = []
zar_values = []
confidence_values = []
price_values = []

since = int(time.time()*1000)-23*60*60*1000
all_trades = []

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

# Fetch BTC/ZAR price history
def fetch_trade_history(pair='XBTZAR'):
    global since, all_trades
    age_limit = int(time.time()*1000)-10*60*60*1000
    res = client.list_trades(pair=pair, since=since)
    all_trades.extend(res['trades'])
    while len(res['trades']) == 100:
        max_timestamp = max(entry['timestamp'] for entry in res['trades'])
        since = max_timestamp + 1
        res = client.list_trades(pair=pair, since=since)
        all_trades.extend(res['trades'])
    
    # Sort all_trades by timestamp
    all_trades.sort(key=lambda x: x['timestamp'])
    
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

# Function to calculate confidence based on slope of order book data
def calculate_slope_confidence(asks, bids, current_price):
    try:
        ask_prices = np.array([float(ask['price']) for ask in asks])
        ask_volumes = np.array([float(ask['volume']) for ask in asks])
        bid_prices = np.array([float(bid['price']) for bid in bids])
        bid_volumes = np.array([float(bid['volume']) for bid in bids])

        # Check if ask_prices or bid_prices are empty
        if len(ask_prices) == 0 or len(bid_prices) == 0:
            return 0.5, 0.5  # or another default value

        cumulative_ask_volumes = np.cumsum(ask_volumes)
        cumulative_bid_volumes = np.cumsum(bid_volumes)

        ask_slope, ask_intercept, _, _, _ = linregress(ask_prices, cumulative_ask_volumes)
        bid_slope, bid_intercept, _, _, _ = linregress(bid_prices, cumulative_bid_volumes)

        # Calculate intercepts at the current price
        asks_intercept_at_current_price = ask_slope * current_price + ask_intercept
        bids_intercept_at_current_price = bid_slope * current_price + bid_intercept

        if ask_slope + (-1 * bid_slope) == 0:
            slope_confidence = 0.5
        else:
            slope_confidence = (-1 * bid_slope) / (ask_slope + (-1 * bid_slope))
        
        if asks_intercept_at_current_price + bids_intercept_at_current_price == 0:
            intercept_confidence = 0.5
        else:
            intercept_confidence = bids_intercept_at_current_price / (asks_intercept_at_current_price + bids_intercept_at_current_price)
            if slope_confidence < 0:
                slope_confidence = 0
            if intercept_confidence < 0:
                intercept_confidence = 0
        return slope_confidence, intercept_confidence
    except:
        return 0.5, 0.5

# Function to calculate price confidence
def calculate_price_confidence():
    try:
        candles = fetch_trade_history()
        df = process_data(candles)
        
        # Calculate the average angle straight line
        start_price = df['Price'].iloc[0]
        end_price = df['Price'].iloc[-1]
        x_diff = (df.index[-1] - df.index[0]).total_seconds()  # Difference in seconds
        y_diff = end_price - start_price

        # Calculate the slope
        slope = y_diff / x_diff
        mapped_value = 1 / (1 + np.exp(-slope))
        return mapped_value
    except Exception as e:
        logging.error(e)
        time.sleep(60)
        return calculate_price_confidence()

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

def get_current_btc_percentage(ticker_data):
    global ZAR_balance, BTC_balance
    # Calculate current BTC value in ZAR
    btc_to_zar = BTC_balance * float(ticker_data['bid'])
    total_value_zar = ZAR_balance + btc_to_zar

    # Calculate current BTC percentage of total value
    current_btc_percentage = 0
    if (total_value_zar != 0):
        current_btc_percentage = btc_to_zar / total_value_zar
    return current_btc_percentage

# Function to determine the action (Buy, Sell, or Nothing) based on confidence
def determine_action(ticker_data, confidence):
    global ZAR_balance, BTC_balance
    # Determine action based on the target confidence and threshold
    btc_to_zar = BTC_balance * float(ticker_data['bid'])
    if confidence > (0.5 + THRESHOLD) and ZAR_balance >  (0.0001 * float(ticker_data['bid'])):
        return 'Buy'
    elif confidence < (0.5 - THRESHOLD) and btc_to_zar > (0.0001 * float(ticker_data['bid'])):
        return 'Sell'
    else:
        return 'Nothing'

# Function to get minimum trade sizes
def get_minimum_trade_sizes():
    return 0.0002  # default value in case of failure

# Update balances by fetching latest balances from the exchange
def update_balances(ticker_data, true_trade, log=False):
    global ZAR_balance, BTC_balance
    try:
        if true_trade:
            balance_response = client.get_balances(assets=['ZAR', 'XBT'])
            while balance_response == None or balance_response['balance'] == None:
                time.sleep(1)
                balance_response = client.get_balances(assets=['ZAR', 'XBT'])
            ZAR_balance = 0
            BTC_balance = 0
            for balance in balance_response['balance']:
                if balance['asset'] == 'ZAR':
                    ZAR_balance += float(balance['balance'])
                elif balance['asset'] == 'XBT':
                    BTC_balance += float(balance['balance'])
        if log:
            logging.info(f'Updated ZAR balance: {ZAR_balance}')
            logging.info(f'Updated BTC balance: {BTC_balance} ({BTC_balance * float(ticker_data["bid"])})')
    except Exception as e:
        logging.error(f'Error fetching updated balances: {e}')

# Function to execute an actual trade
def execute_trade(order_type, ticker_data):
    global ZAR_balance, BTC_balance
    price = float(ticker_data['bid'])
    min_trade_size = get_minimum_trade_sizes()
    logging.info(f"BTC Price: R {price}")
    if order_type == 'Buy':
        try:
            amount = max(float(ZAR_balance/price), min_trade_size)
            client.post_market_order(pair=PAIR, type='BUY', counter_volume=amount)
            logging.info(f'Bought {amount} BTC at {price} ZAR/BTC')
        except Exception as e:
            logging.error(f'Error executing buy order: {e}')
    elif order_type == 'Sell':
        try:
            amount = max(float(BTC_balance), min_trade_size)
            client.post_market_order(pair=PAIR, type='SELL', base_volume=BTC_balance)
            logging.info(f'Sold {amount} BTC at {price} ZAR/BTC')
        except Exception as e:
            logging.error(f'Error executing sell order: {e}')
    update_balances(ticker_data, True)
    current_btc_percentage = get_current_btc_percentage(ticker_data)
    logging.info(f'BTC wallet %: {current_btc_percentage}')

# Mock trade function to print what would happen in a trade
def mock_trade(order_type, ticker_data, fee_info):
    global ZAR_balance, BTC_balance
    logging.info(f"=================================")
    logging.info(f"BTC Price: R {float(ticker_data['bid'])}")
    if order_type == 'Buy':
        amount = max(float(ZAR_balance/price))
        price = float(ticker_data['ask'])
        fee_percentage = float(fee_info['taker_fee'])
        ZAR_balance = 0
        BTC_balance += amount * (1-fee_percentage) 
        logging.info(f'Bought {amount} BTC at {price} ZAR/BTC')
    elif order_type == 'Sell':
        price = float(ticker_data['bid'])
        fee_percentage = float(fee_info['taker_fee'])
        revenue = price * BTC_balance
        fee = revenue * fee_percentage
        total_revenue = revenue - fee
        BTC_balance = 0
        ZAR_balance += total_revenue
        logging.info(f'Sold {amount} BTC at {price} ZAR/BTC')
    logging.info(f'New ZAR balance: {ZAR_balance}')
    logging.info(f'New BTC balance: {BTC_balance} ({BTC_balance * float(ticker_data["bid"])})')
    current_btc_percentage = get_current_btc_percentage(ticker_data)
    logging.info(f'BTC wallet %: {current_btc_percentage}')

# Function to update wallet values for plotting
def update_wallet_values(ticker_data, confidence):
    global ZAR_balance, BTC_balance, time_steps, wallet_values, btc_values_in_zar, zar_values, confidence_values, price_values

    # Calculate current BTC value in ZAR
    btc_to_zar = BTC_balance * float(ticker_data['bid'])
    total_value_zar = ZAR_balance + btc_to_zar

    # Store the current time, wallet value in ZAR, confidence, and price
    current_time = time.time()
    current_price = float(ticker_data['bid'])  # Use 'bid' price as current price
    time_steps.append(current_time)
    wallet_values.append(total_value_zar)
    btc_values_in_zar.append(btc_to_zar)
    zar_values.append(ZAR_balance)
    confidence_values.append(confidence)
    price_values.append(current_price)

    # Ensure all lists are the same length
    min_length = min(len(time_steps), len(wallet_values), len(btc_values_in_zar), len(zar_values), len(confidence_values), len(price_values))
    time_steps = time_steps[:min_length]
    wallet_values = wallet_values[:min_length]
    btc_values_in_zar = btc_values_in_zar[:min_length]
    zar_values = zar_values[:min_length]
    confidence_values = confidence_values[:min_length]
    price_values = price_values[:min_length]

def format_large_number(x, pos):
    if x >= 1e6:
        return f'{x/1e6:.3f}M'
    elif x >= 1e3:
        return f'{x/1e3:.3f}K'
    else:
        return f'{x:.0f}'

# Function to plot wallet values over time
def plot_wallet_values():
    # Convert Unix timestamps to pandas datetime
    time_labels = pd.to_datetime(time_steps, unit='s')

    plt.plot(time_labels, wallet_values, label='Wallet Value in ZAR')
    plt.xlabel('Time')
    plt.ylabel('Wallet Value (ZAR)')
    plt.title('Wallet Value Over Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

# Function to update the plot
def update_plot(frame):
    try:
        fig.clear()
        
        # Create three subplots
        ax1 = fig.add_subplot(311)  # Top subplot for wallet values
        ax2 = fig.add_subplot(312)  # Middle subplot for confidence
        ax3 = fig.add_subplot(313)  # Bottom subplot for price

        min_length = min(len(time_steps), len(wallet_values), len(btc_values_in_zar), len(zar_values))
        
        time_labels = pd.to_datetime(time_steps[:min_length], unit='s')
        
        # Plot wallet values on the top subplot
        ax1.plot(time_labels, wallet_values[:min_length], label='Total Wallet Value in ZAR')
        ax1.plot(time_labels, btc_values_in_zar[:min_length], label='BTC Value in ZAR')
        ax1.plot(time_labels, zar_values[:min_length], label='ZAR Value')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value (ZAR)')
        ax1.set_title('Wallet Values Over Time')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        ax1.yaxis.set_major_formatter(FuncFormatter(format_large_number))
        
        # Plot confidence on the middle subplot
        current_confidence = confidence_values[-1] if confidence_values else 0
        ax2.plot(time_labels, confidence_values, label=f'Confidence ({current_confidence:.2f})', color='purple')
        ax2.axhline(y=0.5, color='r', linestyle='--', label='Midpoint (0.5)')  # Add dashed middle line
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Confidence Over Time')
        ax2.set_ylim(0, 1)  # Set y-axis limits for confidence (0 to 1)
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot price on the bottom subplot
        ax3.plot(time_labels, price_values, label='BTC Price', color='green')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Price (ZAR)')
        ax3.set_title('BTC Price Over Time')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        ax3.yaxis.set_major_formatter(FuncFormatter(format_large_number))
        
        plt.tight_layout()  # Adjust the layout to prevent overlap
    except Exception as e:
        logging.error(f"Error updating plot: {e}")

# Graceful shutdown handler
def signal_handler(sig, frame):
    logging.info('Gracefully shutting down...')
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Main function to check the ticker and place orders
def trading_loop(true_trade):
    global ZAR_balance, BTC_balance
    old_confidence = None
    ticker_data = get_ticker()
    update_balances(ticker_data, true_trade)

    while True:
        fee_info = get_fee_info()
        if not fee_info:
            logging.error('Failed to retrieve fee information')
            continue

        ticker_data = get_ticker()
        if not ticker_data:
            logging.error('Failed to retrieve ticker data')
            continue
        update_balances(ticker_data, true_trade)

        order_book = get_order_book()
        if not order_book:
            logging.error('Failed to retrieve order book')
            continue  # Skip the rest of the loop iteration and try again

        conf_delta = 0
        confidence = calculate_price_confidence()
        if old_confidence is not None:
            conf_delta = confidence - old_confidence 
        old_confidence = confidence

        action = determine_action(ticker_data, confidence)

        # if abs(conf_delta) > 0.01 or action == 'Buy' or action == 'Sell':
        btc_to_zar = BTC_balance * float(ticker_data['bid'])
        total_value_zar = ZAR_balance + btc_to_zar
        current_btc_percentage = btc_to_zar / total_value_zar
        logging.info(f"---------------------------------")
        logging.info(f"BTC Price: R {float(ticker_data['bid'])}")
        logging.info(f'BTC wallet %: {current_btc_percentage}')
        logging.info(f'Confidence in BTC: {confidence}')
        logging.info(f'Confidence Delta: {conf_delta}')

        if action == 'Buy':
            if true_trade:
                execute_trade('Buy', ticker_data)
            else:
                mock_trade('Buy', ticker_data, fee_info)
        elif action == 'Sell':
            if true_trade:
                execute_trade('Sell', ticker_data)
            else:
                mock_trade('Sell', ticker_data, fee_info)
        update_wallet_values(ticker_data, confidence)

        time.sleep(5)

# Argument parser
parser = argparse.ArgumentParser(description='Luno Trading Bot')
parser.add_argument('--true-trade', action='store_true', help='Execute real trades')
args = parser.parse_args()

# Start the trading loop in a separate thread
trading_thread = threading.Thread(target=trading_loop, args=(args.true_trade,))
trading_thread.daemon = True
trading_thread.start()

if __name__ == '__main__':
    fig = plt.figure(figsize=(12, 15))
    ani = FuncAnimation(fig, update_plot, interval=5000, cache_frame_data=False)  # Update every second
    plt.show()  # Show the plot
