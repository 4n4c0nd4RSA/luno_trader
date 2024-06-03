# Luno BTC/ZAR Trader

This project is a simple trading bot for the Luno cryptocurrency exchange, designed to monitor the BTC/ZAR trading pair and make trading decisions based on historical order book data. The bot calculates a confidence level and adjusts the BTC/ZAR ratio in the wallet accordingly. It includes features to record order book history, determine trading actions, and visualize wallet value over time.

## Features
- Fetches and stores the latest order book data.
- Calculates trading confidence based on historical order book data.
- Makes trading decisions (buy, sell, or hold) based on confidence levels.
- Updates wallet values and visualizes them in real-time.
- Records trading actions and updates wallet balances.

## Requirements
- Python 3.x
- luno-python (Luno API client)
- matplotlib (for visualization)
- json (for handling order book history)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/luno-btc-zar-trader.git
    cd luno-btc-zar-trader
    ```

2. Install the required packages:
    ```bash
    pip install luno-python matplotlib
    ```

3. Set up your Luno API credentials by modifying the `API_KEY` and `API_SECRET` constants in the script:
    ```python
    API_KEY = 'your_api_key'
    API_SECRET = 'your_api_secret'
    ```

## Usage
Run the script to start the trading bot:
```bash
python luno_btc_zar_trader.py
```

The bot will:
1. Fetch the latest ticker and order book data.
2. Calculate the confidence level based on historical order book data.
3. Determine the trading action (buy, sell, or hold) based on the confidence level.
4. Update wallet balances and visualize the wallet value over time.

## Project Structure
- `luno_btc_zar_trader.py`: The main script containing the trading bot logic.
- `order_book_history.json`: A JSON file to store order book history data.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```markdown
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgments
- [Luno API](https://www.luno.com/en/developers/api) for providing the cryptocurrency trading platform and API.

## Disclaimer
This project is for educational purposes only. Trading cryptocurrencies carries a high level of risk, and you should carefully consider your investment objectives, level of experience, and risk appetite before engaging in such activities. Use this bot at your own risk.
