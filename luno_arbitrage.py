import os
import logging
from decimal import Decimal
from typing import Dict, List, Tuple, Optional
from luno_python.client import Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = os.getenv('LUNO_API_KEY_ID')
API_SECRET = os.getenv('LUNO_API_KEY_SECRET')

SA_PAIRS = [
    "XBTZAR", "ETHZAR", "LTCZAR", "USDCZAR", "XRPZAR",
    "ETHXBT", "LTCXBT", "BCHXBT", "XRPXBT"
]

def split_pair(pair: str) -> Tuple[str, str]:
    for i in range(3, len(pair) - 2):  # Assuming minimum currency code length is 3
        if pair[:i] in ["XBT", "ETH", "LTC", "USDC", "XRP", "BCH"] and \
           pair[i:] in ["ZAR", "XBT"]:
            return pair[:i], pair[i:]
    raise ValueError(f"Invalid pair: {pair}")

class ArbitrageChecker:
    def __init__(self, api_key: str, api_secret: str, fee: Decimal = Decimal('0.001')):
        self.client = Client(api_key_id=api_key, api_key_secret=api_secret)
        self.pairs = SA_PAIRS
        self.tickers = {}
        self.fee = fee

    def set_fee(self, fee: Decimal):
        self.fee = fee

    def get_ticker(self, pair: str) -> Optional[Dict]:
        if pair not in self.tickers:
            try:
                self.tickers[pair] = self.client.get_ticker(pair=pair)
            except Exception as e:
                logger.error(f"Error fetching {pair}: {e}")
                return None
        return self.tickers[pair]

    def get_all_tickers(self):
        for pair in self.pairs:
            self.get_ticker(pair)

    def calculate_arbitrage(self, path: List[str]) -> Optional[Decimal]:
        tickers = [self.get_ticker(pair) for pair in path]
        if not all(tickers):
            return None

        try:
            amount = Decimal('1')
            current_currency, _ = split_pair(path[0])
            logger.info(f"Starting amount: {amount} {current_currency}")
            logger.info(f"Current trading fee: {self.fee:%}")

            for pair, ticker in zip(path, tickers):
                base, quote = split_pair(pair)
                
                if current_currency == base:
                    # Selling base currency
                    rate = Decimal(ticker['bid'])
                    amount *= rate
                    amount *= (1 - self.fee)  # Apply fee
                    current_currency = quote
                    logger.info(f"Sell {base} for {quote}: {amount} {current_currency} (Rate: {rate}, After fee: {amount})")
                elif current_currency == quote:
                    # Buying base currency
                    rate = Decimal(ticker['ask'])
                    amount /= rate
                    amount *= (1 - self.fee)  # Apply fee
                    current_currency = base
                    logger.info(f"Buy {base} with {quote}: {amount} {current_currency} (Rate: {rate}, After fee: {amount})")
                else:
                    logger.error(f"Invalid path: {path}")
                    return None

            profit_percentage = (amount - Decimal('1')) * 100
            logger.info(f"Final amount: {amount} {current_currency}")
            logger.info(f"Profit percentage: {profit_percentage}%")

            return profit_percentage
        except (KeyError, TypeError, ZeroDivisionError) as e:
            logger.error(f"Error calculating arbitrage for {path}: {e}")
            return None

    def generate_valid_paths(self) -> List[List[str]]:
        valid_paths = []
        for first_pair in self.pairs:
            for second_pair in self.pairs:
                if first_pair == second_pair:
                    continue
                for third_pair in self.pairs:
                    if third_pair == first_pair or third_pair == second_pair:
                        continue

                    # Split the pairs into base and quote currencies
                    first_base, first_quote = split_pair(first_pair)
                    second_base, second_quote = split_pair(second_pair)
                    third_base, third_quote = split_pair(third_pair)

                    # Check if the pairs form a valid path
                    if (first_quote == second_base and second_quote == third_base and third_quote == first_base) or \
                    (first_quote == second_base and second_quote == third_quote and third_base == first_base) or \
                    (first_quote == second_quote and second_base == third_base and third_quote == first_base):
                        valid_paths.append([first_pair, second_pair, third_pair])
        
        return valid_paths

    def check_all_arbitrage(self):
        self.get_all_tickers()
        
        valid_paths = self.generate_valid_paths()
        logger.info(f"Generated {len(valid_paths)} valid paths")
        
        arbitrage_opportunities = []

        for path in valid_paths:
            arbitrage_percentage = self.calculate_arbitrage(path)
            if arbitrage_percentage is not None and arbitrage_percentage > 0:
                arbitrage_opportunities.append((path, arbitrage_percentage))

        arbitrage_opportunities.sort(key=lambda x: x[1], reverse=True)

        for i, (path, percentage) in enumerate(arbitrage_opportunities[:10], 1):
            logger.info(f"{i}. Path: {' -> '.join(path)}, Arbitrage: {percentage:.2f}%")

        if not arbitrage_opportunities:
            logger.info("No arbitrage opportunities found.")

def main():
    if not API_KEY or not API_SECRET:
        logger.error("API credentials not found. Please set LUNO_API_KEY_ID and LUNO_API_KEY_SECRET environment variables.")
        return

    # Initialize with the highest fee tier (0.1%) as a conservative estimate
    arbitrage_checker = ArbitrageChecker(API_KEY, API_SECRET, fee=Decimal('0.001'))
    
    # Alternatively, you can set the fee based on your known tier:
    # arbitrage_checker.set_fee(Decimal('0.0002'))  # Example: 0.02% fee
    
    arbitrage_checker.check_all_arbitrage()

if __name__ == "__main__":
    main()