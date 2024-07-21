import time
import os
import logging
import signal
import sys
import threading
import argparse
import queue
import tzlocal
import luno_python.client as luno
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter
from scipy.stats import linregress
from config import API_CALL_DELAY,PAIR,PERIOD,SHORT_PERIOD,PRICE_CONFIDENCE_THRESHOLD,MARKET_PERCEPTION_THRESHOLD, MARKET_MOMENTUM_INDICATOR_THRESHOLD

# Constants
API_KEY = os.getenv('LUNO_API_KEY_ID')
API_SECRET = os.getenv('LUNO_API_KEY_SECRET')
VERSION = '1.0.2'

# start time
start_time = time.gmtime()

# Initialize logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Initialize the Luno API client
client = luno.Client(api_key_id=API_KEY, api_key_secret=API_SECRET)

# Initialize wallet balances
ZAR_balance = 2000  # Initial ZAR balance
BTC_balance = 0  # Initial PAIR balance

# Lists to store wallet values over time
time_steps = []
wallet_values = []
btc_values_in_zar = []
zar_values = []
confidence_values = []
short_confidence_values = []
market_momentum_indicator_values = []
price_values = []

since = int(time.time()*1000)-23*60*60*1000
all_trades = []

data_queue = queue.Queue()

def extract_base_currency(pair=PAIR):
    if pair.endswith('ZAR'):
        return pair[:-3]
    else:
        return None

def on_home_clicked(event):
    print("'Home' button was clicked!")

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

# Fetch PAIR/ZAR price history
def fetch_trade_history(pair=PAIR):
    global since, all_trades
    age_limit = int(time.time()*1000)-PERIOD*60*60*1000
    short_age_limit = int(time.time()*1000)-SHORT_PERIOD*60*60*1000
    res = client.list_trades(pair=pair, since=since)
    all_trades.extend(res['trades'])
    while len(res['trades']) == 100:
        max_timestamp = max(entry['timestamp'] for entry in res['trades'])
        since = max_timestamp + 1
        res = client.list_trades(pair=pair, since=since)
        all_trades.extend(res['trades'])
    
    # Sort all_trades by timestamp
    all_trades.sort(key=lambda x: x['timestamp'])
    
    all_trades = [trade for trade in all_trades if trade['timestamp'] >= age_limit]
    recent_trades = [trade for trade in all_trades if trade['timestamp'] >= age_limit and trade['timestamp'] <= short_age_limit]
    more_recent_trades = [trade for trade in all_trades if trade['timestamp'] >= short_age_limit]
    return recent_trades, more_recent_trades, all_trades

# Process data into a DataFrame
def process_data(candles):
    data = []
    for candle in candles:
        local_tz = tzlocal.get_localzone()
        timestamp = pd.to_datetime(candle['timestamp'], unit='ms').tz_localize('UTC').tz_convert(local_tz)
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

# Calculate the slope
def calculate_price_slope(df):
    # Calculate the average angle straight line
    start_price = df['Price'].iloc[0]
    end_price = df['Price'].iloc[-1]
    x_diff = (df.index[-1] - df.index[0]).total_seconds()  # Difference in seconds
    y_diff = end_price - start_price

    # Calculate the slope
    slope = y_diff / x_diff
    mapped_value = 1 / (1 + np.exp(-slope))
    return 1-mapped_value

# Function to calculate price confidence
def calculate_price_confidence():
    try:
        candles, more_recent_candles, all = fetch_trade_history(PAIR)
        df = process_data(candles)
        df_short = process_data(more_recent_candles)
        return calculate_price_slope(df), 1-calculate_price_slope(df_short)
    except Exception as e:
        logging.error(e)
        time.sleep(API_CALL_DELAY)
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
    # Calculate current PAIR value in ZAR
    btc_to_zar = BTC_balance * float(ticker_data['bid'])
    total_value_zar = ZAR_balance + btc_to_zar

    # Calculate current PAIR percentage of total value
    current_btc_percentage = 0
    if (total_value_zar != 0):
        current_btc_percentage = btc_to_zar / total_value_zar
    return current_btc_percentage

# Function to determine the action (Buy, Sell, or Nothing) based on confidence
def determine_action(ticker_data, confidence, short_confidence):
    global ZAR_balance, BTC_balance
    # Determine action based on the target confidence and threshold
    mmi = (confidence + short_confidence) / 2
    btc_to_zar = BTC_balance * float(ticker_data['bid'])
    if (confidence >= (0.5 + MARKET_PERCEPTION_THRESHOLD) 
        and short_confidence >= (0.5 + PRICE_CONFIDENCE_THRESHOLD)
        or short_confidence >= (0.5 + PRICE_CONFIDENCE_THRESHOLD) 
        and mmi >= (0.5 + MARKET_MOMENTUM_INDICATOR_THRESHOLD)) and ZAR_balance >  (0.0001 * float(ticker_data['bid'])):
        return 'Buy'
    elif (confidence <= (0.5 - MARKET_PERCEPTION_THRESHOLD) 
          and short_confidence <= (0.5 - PRICE_CONFIDENCE_THRESHOLD) 
          or short_confidence <= (0.5 - PRICE_CONFIDENCE_THRESHOLD) 
          and mmi <= (0.5 - MARKET_MOMENTUM_INDICATOR_THRESHOLD)) and btc_to_zar > (0.0001 * float(ticker_data['bid'])):
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
            balance_response = client.get_balances(assets=['ZAR', extract_base_currency(PAIR)])
            while balance_response == None or balance_response['balance'] == None:
                time.sleep(1)
                balance_response = client.get_balances(assets=['ZAR', extract_base_currency(PAIR)])
            ZAR_balance = 0
            BTC_balance = 0
            for balance in balance_response['balance']:
                if balance['asset'] == 'ZAR':
                    ZAR_balance += float(balance['balance'])
                elif balance['asset'] == extract_base_currency(PAIR):
                    BTC_balance += float(balance['balance'])
        if log:
            logging.info(f'Updated ZAR balance: {ZAR_balance}')
            logging.info(f'Updated {extract_base_currency(PAIR)} balance: {BTC_balance} ({BTC_balance * float(ticker_data["bid"])})')
    except Exception as e:
        logging.error(f'Error fetching updated balances: {e}')

# Function to execute an actual trade
def execute_trade(order_type, ticker_data, fee_info):
    global ZAR_balance, BTC_balance
    price = float(ticker_data['bid'])
    taker_fee_percentage = float(fee_info['taker_fee'])
    logging.info(f"{extract_base_currency(PAIR)} Price: R {price}")
    if order_type == 'Buy':
        try:
            amount = round(float(ZAR_balance-0.01),2)
            logging.info(f'Trying to Buy R{amount} of {extract_base_currency(PAIR)} at {price} ZAR/{extract_base_currency(PAIR)}')
            client.post_market_order(pair=PAIR, type='BUY', counter_volume=amount)
            logging.info(f'Bought {amount} {extract_base_currency(PAIR)} at {price} ZAR/{extract_base_currency(PAIR)}')
        except Exception as e:
            logging.error(f'Error executing buy order: {e}')
    elif order_type == 'Sell':
        try:
            amount = round(float(BTC_balance*(1-taker_fee_percentage))-0.000001,6)
            logging.info(f'Trying to Sell {amount} {extract_base_currency(PAIR)} at {price} ZAR/{extract_base_currency(PAIR)}')
            client.post_market_order(pair=PAIR, type='SELL', base_volume=amount)
            logging.info(f'Sold {amount} {extract_base_currency(PAIR)} at {price} ZAR/{extract_base_currency(PAIR)}')
        except Exception as e:
            logging.error(f'Error executing sell order: {e}')
    update_balances(ticker_data, True)
    current_btc_percentage = get_current_btc_percentage(ticker_data)
    logging.info(f'{extract_base_currency(PAIR)} wallet %: {current_btc_percentage}')
    logging.info(f'New ZAR balance: {ZAR_balance}')
    logging.info(f'New {extract_base_currency(PAIR)} balance: {BTC_balance} ({BTC_balance * float(ticker_data["bid"])})')

# Mock trade function to print what would happen in a trade
def mock_trade(order_type, ticker_data, fee_info):
    global ZAR_balance, BTC_balance
    logging.info(f"=================================")
    logging.info(f"{extract_base_currency(PAIR)} Price: R {float(ticker_data['bid'])}")
    min_trade_size = get_minimum_trade_sizes()
    if order_type == 'Buy':
        price = float(ticker_data['ask'])
        amount = max(float(ZAR_balance/price),min_trade_size)
        fee_percentage = float(fee_info['taker_fee'])
        ZAR_balance = 0
        BTC_balance += amount * (1-fee_percentage) 
        logging.info(f'Bought {amount} {extract_base_currency(PAIR)} at {price} ZAR/{extract_base_currency(PAIR)}')
    elif order_type == 'Sell':
        price = float(ticker_data['bid'])
        fee_percentage = float(fee_info['taker_fee'])
        revenue = price * BTC_balance
        fee = revenue * fee_percentage
        total_revenue = revenue - fee
        BTC_balance = 0
        ZAR_balance += total_revenue
        logging.info(f'Sold {BTC_balance} {extract_base_currency(PAIR)} at {price} ZAR/{extract_base_currency(PAIR)}')
    logging.info(f'New Demo ZAR balance: {ZAR_balance}')
    logging.info(f'New Demo {extract_base_currency(PAIR)} balance: {BTC_balance} ({BTC_balance * float(ticker_data["bid"])})')
    current_btc_percentage = get_current_btc_percentage(ticker_data)
    logging.info(f'{extract_base_currency(PAIR)} wallet %: {current_btc_percentage}')

# Function to update wallet values for plotting
def update_wallet_values(ticker_data, confidence, short_confidence):
    global ZAR_balance, BTC_balance, data_queue

    btc_to_zar = BTC_balance * float(ticker_data['bid'])
    total_value_zar = ZAR_balance + btc_to_zar
    current_time = time.time()
    current_price = float(ticker_data['bid'])
    
    data_queue.put({
        'time': current_time,
        'wallet_value': total_value_zar,
        'btc_value_in_zar': btc_to_zar,
        'zar_value': ZAR_balance,
        'confidence': confidence,
        'short_confidence': short_confidence,
        'price': current_price
    })

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
    local_tz = tzlocal.get_localzone()
    time_labels = pd.to_datetime(time_steps, unit='s').tz_localize('UTC').tz_convert(local_tz)

    plt.plot(time_labels, wallet_values, label='Wallet Value in ZAR')
    plt.xlabel('Time')
    plt.ylabel('Wallet Value (ZAR)')
    plt.title('Wallet Value Over Time')
    plt.xticks(rotation=45)
    plt.legend(loc='center left', bbox_to_anchor=(0, 0.5))
    plt.show()

# Function to update the plot
def update_plot(frame):
    global time_steps, wallet_values, btc_values_in_zar, zar_values, confidence_values, short_confidence_values, market_momentum_indicator_values, price_values, fig, ax1, ax2, ax3, data_queue
    try:
        while not data_queue.empty():
            data = data_queue.get_nowait()
            time_steps.append(data['time'])
            wallet_values.append(data['wallet_value'])
            btc_values_in_zar.append(data['btc_value_in_zar'])
            zar_values.append(data['zar_value'])
            confidence_values.append(data['confidence'])
            short_confidence_values.append(data['short_confidence'])
            market_momentum_indicator_values.append((data['confidence'] + data['short_confidence'])/2)
            price_values.append(data['price'])

        # Ensure all lists are the same length
        min_length = min(len(time_steps), len(wallet_values), len(btc_values_in_zar), len(zar_values), len(confidence_values), len(short_confidence_values), len(price_values), len(market_momentum_indicator_values))
        time_steps = time_steps[-min_length:]
        wallet_values = wallet_values[-min_length:]
        btc_values_in_zar = btc_values_in_zar[-min_length:]
        zar_values = zar_values[-min_length:]
        confidence_values = confidence_values[-min_length:]
        short_confidence_values = short_confidence_values[-min_length:]
        market_momentum_indicator_values = market_momentum_indicator_values[-min_length:]
        price_values = price_values[-min_length:]

        ax1.clear()
        ax2.clear()
        ax3.clear()

        min_length = min(len(time_steps), len(wallet_values), len(btc_values_in_zar), len(zar_values))
        
        local_tz = tzlocal.get_localzone()
        time_labels = pd.to_datetime(time_steps[:min_length], unit='s').tz_localize('UTC').tz_convert(local_tz)
        
        # Plot wallet values on the top subplot
        ax1.plot(time_labels, wallet_values[:min_length], label='Total Wallet Value in ZAR')
        ax1.plot(time_labels, btc_values_in_zar[:min_length], label=f'{extract_base_currency(PAIR)} Value in ZAR')
        ax1.plot(time_labels, zar_values[:min_length], label='ZAR Value')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value (ZAR)')
        ax1.set_title('Wallet Values Over Time')
        ax1.legend(loc='center left', bbox_to_anchor=(0, 0.5))
        ax1.tick_params(axis='x', rotation=45)
        ax1.yaxis.set_major_formatter(FuncFormatter(format_large_number))
        
        # Plot confidence on the middle subplot
        current_confidence = confidence_values[-1] if confidence_values else 0
        current_short_confidence = short_confidence_values[-1] if short_confidence_values else 0
        current_mmi = market_momentum_indicator_values[-1] if market_momentum_indicator_values else 0
        ax2.plot(time_labels, short_confidence_values, label=f'Price Confidence ({current_short_confidence:.2f})', color='#c76eff')
        ax2.plot(time_labels, market_momentum_indicator_values, label=f'Market Momentum Indicator ({current_mmi:.2f})', color='#f51bbe')
        ax2.plot(time_labels, confidence_values, label=f'Market Perception ({current_confidence:.2f})', color='purple')
        ax2.axhline(y=0.5 + PRICE_CONFIDENCE_THRESHOLD, color='g', linestyle='--', label=f'PC Buy Limit ({0.5 + PRICE_CONFIDENCE_THRESHOLD})')
        ax2.axhline(y=0.5 + MARKET_MOMENTUM_INDICATOR_THRESHOLD, color='#0db542', linestyle='--', label=f'MMI Buy Limit ({0.5 + MARKET_MOMENTUM_INDICATOR_THRESHOLD})')
        ax2.axhline(y=0.5 + MARKET_PERCEPTION_THRESHOLD, color='lime', linestyle='--', label=f'MP Buy Limit ({0.5 + MARKET_PERCEPTION_THRESHOLD})')
        ax2.axhline(y=0.5, color='black', linestyle='--', label='Midpoint (0.5)')
        ax2.axhline(y=0.5 - MARKET_PERCEPTION_THRESHOLD, color='pink', linestyle='--', label=f'MP Sell Limit ({0.5 - MARKET_PERCEPTION_THRESHOLD})')
        ax2.axhline(y=0.5 - MARKET_MOMENTUM_INDICATOR_THRESHOLD, color='#f5b8cf', linestyle='--', label=f'MMI Sell Limit ({0.5 - MARKET_MOMENTUM_INDICATOR_THRESHOLD})')
        ax2.axhline(y=0.5 - PRICE_CONFIDENCE_THRESHOLD, color='r', linestyle='--', label=f'PC Sell Limit ({0.5 - PRICE_CONFIDENCE_THRESHOLD})')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price Confidence')
        ax2.set_title('Market Indicators')
        ax2.set_ylim(0, 1)  # Set y-axis limits for confidence (0 to 1)
        ax2.legend(loc='center left', bbox_to_anchor=(0, 0.5))
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot price on the bottom subplot
        ax3.plot(time_labels, price_values, label=f'{extract_base_currency(PAIR)} Price', color='green')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Price (ZAR)')
        ax3.set_title(f'{extract_base_currency(PAIR)} Price Over Time')
        ax3.legend(loc='center left', bbox_to_anchor=(0, 0.5))
        ax3.tick_params(axis='x', rotation=45)
        ax3.yaxis.set_major_formatter(FuncFormatter(format_large_number))

        ax1.autoscale_view(scalex=True, scaley=True)
        ax2.autoscale_view(scalex=True, scaley=True)
        ax3.autoscale_view(scalex=True, scaley=True)
        
        plt.tight_layout()  # Adjust the layout to prevent overlap
    except Exception as e:
        logging.error(f"Error updating plot: {e}")

# Graceful shutdown handler
def signal_handler(sig, frame):
    logging.info('Gracefully shutting down...')
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def trading_loop(true_trade):
    global ZAR_balance, BTC_balance
    old_confidence = None

    while True:
        try:
            ticker_data = get_ticker()
            if not ticker_data:
                logging.error('Failed to retrieve ticker data')
                time.sleep(API_CALL_DELAY)
                continue

            update_balances(ticker_data, true_trade)

            order_book = get_order_book()
            if not order_book:
                logging.error('Failed to retrieve order book')
                time.sleep(API_CALL_DELAY)
                continue

            confidence, short_confidence = calculate_price_confidence()
            conf_delta = 0
            if old_confidence is not None:
                conf_delta = confidence - old_confidence 
            old_confidence = confidence

            action = determine_action(ticker_data, confidence, short_confidence)

            if abs(conf_delta) > 0.01 or action in ['Buy', 'Sell']:
                logging.info(f"---------------------------------")
                logging.info(f"{extract_base_currency(PAIR)} Price: R {float(ticker_data['bid'])}")
                logging.info(f'{extract_base_currency(PAIR)} wallet %: {get_current_btc_percentage(ticker_data)}')
                logging.info(f'Market Perception in {extract_base_currency(PAIR)}: {confidence}')
                logging.info(f'Market Perception Delta: {conf_delta}')
                logging.info(f'Price Confidence: {short_confidence}')
            if action in ['Buy', 'Sell']:
                fee_info = get_fee_info()
                if not fee_info:
                    logging.error('Failed to retrieve fee information')
                    time.sleep(API_CALL_DELAY)
                    continue

                if true_trade:
                    execute_trade(action, ticker_data, fee_info)
                else:
                    mock_trade(action, ticker_data, fee_info)

            update_wallet_values(ticker_data, confidence, short_confidence)

        except Exception as e:
            logging.error(f"Error in trading loop: {e}")
        finally:
            time.sleep(API_CALL_DELAY)

if __name__ == '__main__':
    # Set up the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    parser = argparse.ArgumentParser(description='Luno Trading Bot')
    parser.add_argument('--true-trade', action='store_true', help='Execute real trades')
    args = parser.parse_args()

    window_title = f'Luno {extract_base_currency(PAIR)}/ZAR Trading Bot - v{VERSION}'
    if args.true_trade != True:
        window_title += ' (Demo Mode)'
    fig.canvas.manager.set_window_title(window_title)

    # Start the trading loop in a separate thread
    trading_thread = threading.Thread(target=trading_loop, args=(args.true_trade,))
    trading_thread.daemon = True
    trading_thread.start()
    ani = FuncAnimation(fig, update_plot, interval=API_CALL_DELAY*1000, cache_frame_data=False)
    plt.show()
