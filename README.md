# Luno BTC/ZAR Trader

This project is a trading bot for the Luno cryptocurrency exchange, designed to monitor the BTC/ZAR trading pair and make trading decisions based on historical order book data. The bot calculates a confidence level and adjusts the BTC/ZAR ratio in the wallet accordingly. It includes features to record order book history, determine trading actions, and visualize various aspects of the trading process.

## Features
- Fetches and analyzes real-time order book data
- Calculates trading confidence based on historical order book data
- Makes trading decisions (buy, sell, or hold) based on confidence levels
- Updates wallet values and visualizes them in real-time
- Records trading actions and updates wallet balances
- Analyzes order book depth and visualizes cumulative order book volumes
- Displays real-time price trends
- Checks for arbitrage opportunities across multiple trading pairs
- Supports both mock trading and real trading modes

## Requirements
- Python 3.9.x
- luno-python (Luno API client)
- matplotlib (for visualization)
- numpy (for numerical operations)
- pandas (for data manipulation)
- scipy (for statistical analysis)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/4n4c0nd4RSA/luno_trader.git
    cd luno_trader
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your Luno API credentials:
   - Set the following environment variables:
     ```bash
     export LUNO_API_KEY_ID=your_api_key
     export LUNO_API_KEY_SECRET=your_api_secret
     ```

## Usage

### Main Trading Bot
Run the script to start the trading bot in MOCK mode:
```bash
python luno_btc_zar_trader.py
```

Run the script to start the trading bot in TRADE mode:
```bash
python luno_btc_zar_trader.py --true-trade
```

The bot will:
1. Fetch the latest ticker and order book data
2. Calculate the confidence level based on historical order book data
3. Determine the trading action (buy, sell, or hold) based on the confidence level
4. Update wallet balances and visualize the wallet value over time

### Additional Scripts
To run other analysis and visualization scripts:

```bash
python luno_confidence.py
python luno_depth.py
python luno_price.py
python luno_arbitrage.py
```

## Project Structure
- `luno_btc_zar_trader.py`: The main script containing the trading bot logic
- `luno_confidence.py`: Script to calculate and visualize confidence levels based on historical order book data
- `luno_depth.py`: Script to analyze order book depth and visualize cumulative order book volumes
- `luno_price.py`: Displays real-time price trends
- `luno_arbitrage.py`: Checks for arbitrage opportunities across multiple trading pairs

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgments
- [Luno API](https://www.luno.com/en/developers/api) for providing the cryptocurrency trading platform and API.

## Disclaimer
This project is for educational purposes only. Trading cryptocurrencies carries a high level of risk, and you should carefully consider your investment objectives, level of experience, and risk appetite before engaging in such activities. The authors are not responsible for any financial losses incurred through the use of this software. Use this bot at your own risk.
