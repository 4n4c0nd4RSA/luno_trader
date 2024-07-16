# Luno ZAR Trader

This project is a trading bot for the Luno cryptocurrency exchange, designed to monitor the CRYPTO/ZAR trading pair and make trading decisions based on historical order book data. The bot calculates a confidence level and adjusts the CRYPTO/ZAR ratio in the wallet accordingly. It includes features to record order book history, determine trading actions, and create graphs of various aspects of the trading process.

## Latest Updates

Version 1.0.1 has been released! This update includes improvements to the Market Perception formula and configuration process. For full details on the changes and new features, please see the [1.0.1 Patch Notes](https://github.com/4n4c0nd4RSA/luno_trader/blob/features/update_notes/1.0.1/1.0.1.md).

## Features
- Fetches and analyzes real-time order book data
- Calculates trading confidence based on historical order book data
- Makes trading decisions (buy, sell, or hold) based on confidence levels
- Updates wallet values and creates graphs of them in real-time
- Records trading actions and updates wallet balances
- Analyzes order book depth and creates graphs of cumulative order book volumes
- Displays real-time price trends through graphing
- Checks for arbitrage opportunities across multiple trading pairs
- Supports both demo trading and real trading modes

## Requirements
You will need a Luno account with an API Key to grant the bot access to your Luno account.
The installation methods below will handle the installation of all necessary components. For advanced users, detailed technical requirements are listed in the `requirements.txt` file.

## Setting Up Your Luno Account

1. Create a Luno account if you don't have one already. [Sign up here](https://www.luno.com/signup)
2. Verify your account following Luno's instructions. [Verification guide](https://www.luno.com/help/en/articles/1000203409)
3. Create API keys for the bot:
   - Log in to your Luno account
   - Go to the [API Keys section](https://www.luno.com/wallet/security/api_keys)
   - Click "Create new key"
   - Give it a name (e.g., "Trading Bot")
   - Select the permissions you want to grant (read balance and trade)
   - Click "Create key"
   - Copy the API Key ID and Secret (you'll need these later)

## Installation

You have two options for obtaining the project files:

### Option 1: Using Git (Recommended for easy updates)

Git is a version control system used by programmers to track changes in code, collaborate with others, and manage different versions of a project. It's especially useful for keeping your local copy of the project up-to-date with the latest changes.

1. Install Git if you haven't already: [Git Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
2. Open a command prompt or terminal
3. Navigate to the directory where you want to install the project
4. Run the following command:
    ```
    git clone https://github.com/4n4c0nd4RSA/luno_trader.git
    cd luno_trader
    ```

To update the project in the future, navigate to the project directory and run:
```
git pull
```

### Option 2: Manual Download

1. Download the project files from [this link](https://github.com/4n4c0nd4RSA/luno_trader/archive/refs/heads/main.zip)
2. Extract the ZIP file to a folder of your choice

Note: With this method, you'll need to manually download and extract the project again for future updates.

After obtaining the project files using either method above, you can proceed with one of the following installation methods:

### Easy Install (Recommended for Windows Users)

1. Navigate to the project folder
2. Double-click on `install.bat` to install all necessary components

Note: The Easy Install method automatically sets up the environment variables for you, so you can skip the "Setting Up Environment Variables" section below.

### Manual Install (For advanced users or non-Windows systems)

1. Open a command prompt or terminal in the project folder
2. Run the following command to install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Setting Up Environment Variables - Manual Setup

This step is only necessary if you did not use the Easy Install method. To securely use your API keys, you need to set them as environment variables:

### Windows
1. Press Win + R, type "sysdm.cpl", and press Enter
2. Go to the "Advanced" tab and click "Environment Variables"
3. Under "User variables", click "New"
4. Add these variables:
   - Variable name: LUNO_API_KEY_ID
     Variable value: your_api_key
   - Variable name: LUNO_API_KEY_SECRET
     Variable value: your_api_secret

### macOS/Linux
Add these lines to your `~/.bash_profile` or `~/.zshrc` file:
```bash
export LUNO_API_KEY_ID=your_api_key
export LUNO_API_KEY_SECRET=your_api_secret
```

Replace `your_api_key` and `your_api_secret` with the values you copied earlier.

## Configuration

The project includes a `config.bat` file for easy configuration of the bot:

1. Navigate to the project folder
2. Double-click on `config.bat`
3. Follow the prompts to set values for:
   - Trading pair (e.g., XBTZAR)
   - Market Perception Threshold
   - Confidence threshold
   - API call delay
   - Period value
   - Short period value
   - Depth chart range

The script will generate a `config.py` file with your chosen settings. If you don't change a value, it will use the default.

## Usage

### Easy Run (Windows Users)

For Windows users, there's a convenient `run.bat` file that simplifies the process of running different components of the project:

1. Navigate to the project folder
2. Double-click on `run.bat`
3. Follow the on-screen prompts to select which component you want to run

This batch file will ask you what you want to run and then execute the appropriate Python script for you.

### Manual Run

If you prefer to run the scripts manually or if you're not using Windows, you can use the following commands:

#### Main Trading Bot
To start the trading bot in demo mode (no real trades):
1. Open a command prompt or terminal in the project folder
2. Run:
    ```
    python luno_zar_trader.py
    ```

To start the bot with real trading (uses actual funds):
```
python luno_zar_trader.py --true-trade
```

The bot will:
1. Get the latest ticker and order book data
2. Calculate the confidence level using past order book data
3. Decide whether to buy, sell, or hold based on the confidence level
4. Update wallet balances and create a graph of the wallet value over time

#### Other Analysis Tools
To run other analysis and graphing tools:

```
python luno_confidence.py
python luno_depth.py
python luno_price.py
python luno_arbitrage.py
```

## Important Disclaimer

This bot trades based on market swing momentum. If you start the bot when the market is at the end of a swing, it might buy too late and then only be ready for buy events in future runs. Always monitor the bot's performance and adjust your strategy as needed.

Trading cryptocurrencies is risky. This project is for educational purposes only. Carefully consider your investment goals, experience, and risk tolerance before trading. The authors are not responsible for any financial losses. Use this bot at your own risk.

This project is for educational purposes only. Trading cryptocurrencies carries a high level of risk, and you should carefully consider your investment objectives, level of experience, and risk appetite before engaging in such activities. The authors are not responsible for any financial losses incurred through the use of this software. Use this bot at your own risk.

## License
This project uses the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
We welcome contributions! Please open an issue or submit a pull request for any improvements or bug fixes.

## More Information
- [Luno API Documentation](https://www.luno.com/en/developers/api)
- [Luno Help Center](https://www.luno.com/help/en/)