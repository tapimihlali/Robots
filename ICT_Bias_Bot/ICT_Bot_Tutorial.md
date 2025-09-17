# ICT Bias Bot Tutorial

## 1. Introduction

Welcome to the ICT Bias Bot! This automated trading bot is designed to implement trading strategies based on concepts from "The Inner Circle Trader" (ICT). The bot determines a daily directional bias and looks for high-probability entry opportunities based on Fair Value Gaps (FVGs) and Market Structure Shifts (MSS).

This tutorial will guide you through the setup, configuration, and operation of the bot in its various modes: backtesting, optimization, and live trading.

## 2. Setup and Installation

Follow these steps to get the bot ready to run on your computer.

### Prerequisites

*   **Python 3:** Make sure you have Python 3 installed on your system. You can download it from [python.org](https://python.org).
*   **MetaTrader 5 (MT5):** The bot requires the MT5 terminal to be installed and running to access market data and execute trades.

### Step 1: Install Required Libraries

The bot uses several Python libraries for data analysis, trading, and notifications. You can install all of them with a single command.

1.  Open a terminal or command prompt.
2.  Navigate to the `ICT_Bias_Bot` folder.
3.  Run the following command to install the required libraries from the `requirements.txt.txt` file:

    ```bash
    pip install -r requirements.txt.txt
    ```

### Step 2: Configure Credentials

The bot needs your MetaTrader 5 account credentials to connect to your broker. These are stored in the `settings.py` file.

1.  Open the `ICT_Bias_Bot/settings.py` file in a text editor.
2.  Update the following variables with your MT5 account details:

    ```python
    # MT5 Credentials
    MT5_USER = 12345678  # Replace with your MT5 account number
    MT5_PASS = "your_password"  # Replace with your MT5 password
    MT5_SERVER = "your_server"  # Replace with your broker's server name
    MT5_PATH = "C:\Program Files\MetaTrader 5\terminal64.exe" # Replace with the path to your MT5 terminal
    ```

### Step 3: Set Up Telegram Notifications (Optional)

The bot can send you real-time notifications about its activities using Telegram.

1.  Open the `ICT_Bias_Bot/config.py` file.
2.  Find the `TELEGRAM_NOTIFICATIONS` section.
3.  Set `TELEGRAM_ENABLED` to `True`.
4.  Replace the placeholder values for `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` with your own.

    ```python
    # --- TELEGRAM NOTIFICATIONS ---
    TELEGRAM_ENABLED = True # Set to False to disable notifications
    TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN" # Replace with your bot token
    TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"       # Replace with your chat ID
    ```

    To get a Bot Token and Chat ID, you will need to:
    1.  Create a new bot by talking to the `@BotFather` on Telegram.
    2.  Get your Chat ID by talking to the `@userinfobot`.

---

## 3. Configuration (`config.py`)

The `config.py` file is the main control panel for the bot. Here you can define the symbols to trade, risk management parameters, and strategy rules.

### Trading Parameters

*   `SYMBOLS`: A list of the financial instruments you want the bot to trade.
*   `DAILY_TIMEFRAME`: The timeframe used to determine the daily bias (default is "D1").
*   `ENTRY_TIMEFRAMES`: The higher timeframes the bot will scan for Fair Value Gaps (default is ["H4", "H1"]).

### Risk Management

*   `SL_BUFFER_PIPS`: A dictionary to set a buffer for the stop loss in pips. You can set a `default` value and override it for specific symbols.
*   `MAX_SL_PIPS`: A dictionary to set the maximum allowed stop loss in pips.
*   `TP_RULE`: The rule for setting the take profit. Can be a fixed risk/reward ratio (e.g., '1:3') or 'PDH/PDL' to target the previous day's high or low.
*   `RISK_PER_TRADE_PERCENT`: The percentage of your equity to risk on a single trade when using the `RISK_PERCENT` lot sizing strategy.
*   `DEFAULT_LOT_SIZE_STRATEGY`: The default method for calculating the trade volume. Can be `MIN_LOT` or `RISK_PERCENT`.
*   `CUSTOM_LOT_SIZES`: A dictionary to set a fixed lot size for specific symbols, overriding the default strategy.
*   `MAX_OPEN_TRADES_GLOBAL`: The maximum number of concurrent trades allowed for any single symbol.
*   `CUSTOM_MAX_OPEN_TRADES`: A dictionary to override the global max open trades for specific symbols.

### Strategy Rules

*   `FVG_LOCATION_RULE`: A dictionary to enable or disable the FVG location rule for each mode (live, backtest, optimizer, walk_forward).
*   `KILLER_ZONE_FILTER`: A dictionary to enable or disable the ICT killer timezone filter for each mode.
*   `ENABLE_INVALIDATION_EXIT`: A dictionary to control the FVG-based invalidation exit logic. You can set a `default` value and override it for specific symbols.
*   `TRAILING_STOP_LOSS`: A dictionary to configure the trailing stop loss settings for live and backtest modes.

---

## 4. Running the Bot

The bot can be run in several modes from your terminal.

### Backtesting

To test the bot's strategy on historical data, run the `backtest.py` script:

```bash
python backtest.py
```

This will generate two files:

*   `ICT_backtest_results_Portfolio.xlsx`: An Excel file with a detailed performance summary and trade history.
*   `Portfolio_backtest_chart.png`: A chart visualizing the equity curve and drawdown.

### Optimization

To find the best strategy parameters for a specific symbol, run the `optimizer.py` script:

```bash
python optimizer.py
```

The optimizer will test a range of parameters defined within the script and save the results to:

*   `ICT_optimization_results.xlsx`: An Excel file ranking the parameter combinations by performance.

### Walk-Forward Optimization

For a more robust optimization, use the `optimizer_walk_forward.py` script. This method tests parameters over different time periods to avoid overfitting.

```bash
python optimizer_walk_forward.py
```

### Live Trading

To run the bot with a live account, execute the `main.py` script:

```bash
python main.py
```

The bot will start running, and you will see its activity in the terminal. You can stop the bot gracefully by pressing `Ctrl+C`.

---

## 5. Understanding the Files

Here is a brief overview of the files in the `ICT_Bias_Bot` directory:

*   `api_client.py`: Handles communication with the MetaTrader 5 terminal for live trading.
*   `backtest.py`: Contains the logic for backtesting the strategy on historical data.
*   `config.py`: The main configuration file for the bot.
*   `ICT_Bias_Bot.ipynb`: A Jupyter notebook for interactive testing and analysis.
*   `invalidation_analyzer.py`: A script to analyze trades that were closed due to invalidation.
*   `main.py`: The entry point for running the bot in live trading mode.
*   `notifier.py`: Handles sending notifications via Telegram.
*   `optimizer.py`: Contains the logic for optimizing strategy parameters.
*   `optimizer_walk_forward.py`: Contains the logic for walk-forward optimization.
*   `requirements.txt.txt`: A list of required Python libraries.
*   `settings.py`: Stores sensitive information like account credentials.
*   `strategy.py`: Contains the core trading logic and strategy rules.
*   `symbol_checker.py`: A utility to check symbol information.

---

## 6. Trading Strategy Explained

The bot's strategy is based on the following ICT concepts:

### Daily Bias

The bot first determines the daily bias (bullish or bearish) by analyzing the daily chart. This sets the direction for all trades for the day.

### Fair Value Gap (FVG)

The bot identifies FVGs on the higher timeframes (`H4`, `H1`). An FVG is a three-candle pattern that indicates an inefficiency or imbalance in the market.

### Market Structure Shift (MSS)

Once the price enters an FVG, the bot looks for an MSS on a lower timeframe. An MSS confirms a change in market structure and provides a potential entry point.

### Entry and Exit

*   **Entry:** A trade is entered when an MSS occurs within an FVG that is aligned with the daily bias.
*   **Stop Loss:** The stop loss is placed below the low of the MSS for a buy trade, or above the high for a sell trade.
*   **Take Profit:** The take profit is determined by the `TP_RULE` in the configuration (e.g., a fixed risk/reward ratio or the previous day's high/low).
*   **Invalidation Exit:** If enabled, the trade will be closed if the price closes beyond the FVG, invalidating the trade setup.
*   **Trailing Stop:** If enabled, the stop loss will be moved to breakeven and then trailed behind the price to lock in profits.
