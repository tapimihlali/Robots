# Import necessary libraries
import argparse  # For parsing command-line arguments
import csv  # For working with CSV files
import logging  # For logging events and errors
import os  # For interacting with the operating system (e.g., checking file existence)
import time  # For adding delays
from datetime import datetime, timedelta  # For working with dates and times

import matplotlib.pyplot as plt  # For creating plots and charts
import pandas as pd  # For data manipulation and analysis
import pandas_ta as ta  # For technical analysis indicators
import pytz  # For handling timezones
from apscheduler.schedulers.blocking import BlockingScheduler  # For scheduling the bot to run at intervals

import MetaTrader5 as mt5  # The official MetaTrader 5 API library
import deriv_mt5_bot_config as config

# --- CONFIGURATION ---
credentials = config.get_credentials()
MT5_USER = credentials['user']
MT5_PASS = credentials['password']
MT5_SERVER = credentials['server']
MT5_PATH = credentials['path']

VIX_INDICES = config.get_symbols()

# --- STRATEGY PARAMETERS ---
TF_TREND = config.TF_TREND
TF_ENTRY = config.TF_ENTRY
EMA_TREND_PERIOD = config.EMA_TREND_PERIOD
EMA_ENTRY_PERIOD = config.EMA_ENTRY_PERIOD

# --- RISK MANAGEMENT SETTINGS ---
SL_MULTIPLIER = config.SL_MULTIPLIER
TP_RISK_REWARD = config.TP_RISK_REWARD

# --- CUSTOM SYMBOL SETTINGS ---
CUSTOM_LOT_SIZES = config.CUSTOM_LOT_SIZES
CUSTOM_MAX_OPEN_TRADES = config.CUSTOM_MAX_OPEN_TRADES
CUSTOM_RISK_PARAMETERS = config.CUSTOM_RISK_PARAMETERS

# --- Bot Configuration and Logging ---
# Configure the logging system to output messages to both a file and the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
                logging.FileHandler(os.path.join(OUTPUT_DIR, "trading_bot_mt5.log")),  # Log messages to a file named trading_bot_mt5.log
        logging.StreamHandler()  # Also output log messages to the console
    ]
)
logger = logging.getLogger(__name__)  # Create a logger instance for the bot

# --- Daily Performance Tracker ---
class DailyPerformanceTracker:
    """Tracks and logs daily performance metrics like drawdown and win rate."""

    def __init__(self, logger_instance):
        """
        Initializes the tracker.
        This function is called when a new DailyPerformanceTracker object is created.
        It sets up the log file, header, and initial stats for the day.
        Called from: TradingBot.__init__
        """
        self.logger = logger_instance  # Use the logger instance from the main bot
        self.log_file = os.path.join(OUTPUT_DIR, 'daily_performance.csv')  # Name of the CSV file for daily stats
        # Define the header for the daily performance CSV file
        self.header = [
            'date', 'peak_equity', 'max_daily_drawdown_pct', 'max_daily_drawdown_dollars',
            'end_of_day_win_rate_pct', 'daily_profit', 'trades_opened',
            'trades_closed', 'sl_hits', 'sl_hit_rate_pct', 'tp_hits',
            'tp_hit_rate_pct'
        ]
        self.today_str = ""  # Initialize today's date string as empty
        self.today_stats = {}  # Initialize today's statistics dictionary as empty
        self._init_log_file()  # Ensure the log file exists and has a header
        self._check_date_rollover(None)  # Check if the date has changed since the last run

    def _init_log_file(self):
        """
        Initializes the daily performance log file.
        If the file doesn't exist, it creates it and writes the header row.
        Called from: __init__
        """
        if not os.path.exists(self.log_file):  # Check if the log file already exists
            with open(self.log_file, 'w', newline='') as f:  # Open the file in write mode
                writer = csv.writer(f)  # Create a CSV writer object
                writer.writerow(self.header)  # Write the header row to the file

    def _load_or_create_today_stats(self):
        """
        Loads today's stats from the CSV file if they exist, otherwise creates a new entry.
        Called from: _check_date_rollover
        """
        try:
            df = pd.read_csv(self.log_file)  # Read the entire CSV file into a pandas DataFrame
            today_data = df[df['date'] == self.today_str]  # Find the row for the current date
            if not today_data.empty:  # If data for today already exists
                self.logger.info(f"Loaded existing performance stats for {self.today_str}.")
                return today_data.iloc[0].to_dict()  # Return the existing data as a dictionary
            else:  # If no data for today exists
                self.logger.info(f"No stats found for {self.today_str}. Creating new entry.")
                # Create a dictionary with default values for a new day
                new_stats = {
                    'date': self.today_str, 'peak_equity': 0.0,
                    'max_daily_drawdown_pct': 0.0, 'max_daily_drawdown_dollars': 0.0,
                    'end_of_day_win_rate_pct': 0.0, 'daily_profit': 0.0, 'trades_opened': 0,
                    'trades_closed': 0, 'sl_hits': 0, 'sl_hit_rate_pct': 0.0, 'tp_hits': 0,
                    'tp_hit_rate_pct': 0.0
                }
                new_df = pd.DataFrame([new_stats])  # Create a new DataFrame for the new stats
                updated_df = pd.concat([df, new_df], ignore_index=True)  # Add the new row to the existing DataFrame
                updated_df.to_csv(self.log_file, index=False)  # Save the updated DataFrame back to the CSV
                return new_stats
        except (FileNotFoundError, pd.errors.EmptyDataError):
            # Handle cases where the file doesn't exist or is empty
            self.logger.warning(f"Could not read {self.log_file}, creating new one.")
            new_stats = {
                'date': self.today_str, 'peak_equity': 0.0,
                'max_daily_drawdown_pct': 0.0, 'max_daily_drawdown_dollars': 0.0,
                'end_of_day_win_rate_pct': 0.0, 'daily_profit': 0.0, 'trades_opened': 0,
                'trades_closed': 0, 'sl_hits': 0, 'sl_hit_rate_pct': 0.0, 'tp_hits': 0,
                'tp_hit_rate_pct': 0.0
            }
            pd.DataFrame([new_stats]).to_csv(self.log_file, index=False)  # Create a new file with the stats
            return new_stats
        except Exception as e:
            self.logger.error(f"Error loading or creating daily stats file: {e}")
            # Return a default dictionary in case of other errors
            return {
                'date': self.today_str, 'peak_equity': 0.0,
                'max_daily_drawdown_pct': 0.0, 'max_daily_drawdown_dollars': 0.0,
                'end_of_day_win_rate_pct': 0.0, 'daily_profit': 0.0, 'trades_opened': 0,
                'trades_closed': 0, 'sl_hits': 0, 'sl_hit_rate_pct': 0.0, 'tp_hits': 0,
                'tp_hit_rate_pct': 0.0
            }

    def _check_date_rollover(self, bot):
        """
        Checks if the current date has changed. If so, resets daily counters.
        Called from: __init__, update, update_daily_profit, update_trade_stats
        """
        current_date_str = datetime.now(pytz.utc).strftime('%Y-%m-%d')  # Get the current date in UTC
        if current_date_str != self.today_str:  # Compare with the last known date
            self.logger.info(f"Date rollover detected. New day: {current_date_str}")
            self.today_str = current_date_str  # Update the date
            self.today_stats = self._load_or_create_today_stats()  # Load or create stats for the new day
            if bot:  # If the bot instance is provided
                bot._reset_daily_counters()  # Reset the bot's daily counters

    def _save_stats(self):
        """
        Saves the current daily statistics to the CSV file.
        Called from: update, update_daily_profit, update_trade_stats
        """
        try:
            df = pd.read_csv(self.log_file)  # Read the CSV file
            idx = df.index[df['date'] == self.today_str].tolist()  # Find the index of today's row
            if idx:  # If today's row exists
                for key, value in self.today_stats.items():  # Loop through the stats dictionary
                    df.loc[idx[0], key] = value  # Update the values in the DataFrame
            else:  # If today's row doesn't exist (should be rare)
                new_df = pd.DataFrame([self.today_stats])  # Create a new DataFrame for the stats
                df = pd.concat([df, new_df], ignore_index=True)  # Add it as a new row
            df.to_csv(self.log_file, index=False)  # Save the changes to the CSV file
        except Exception as e:
            self.logger.error(f"Failed to save daily performance stats: {e}")

    def update(self, current_equity, win_rate, bot):
        """
        Updates daily performance metrics like peak equity and drawdown.
        Called from: TradingBot.run_bot_iteration
        """
        self._check_date_rollover(bot)  # Check for date change

        # Update peak equity for the day
        peak_equity = self.today_stats.get('peak_equity', 0.0)
        if peak_equity == 0.0 and current_equity > 0:
            self.logger.info(f"Setting initial peak equity for the day: {current_equity:.2f}")
            self.today_stats['peak_equity'] = current_equity

        if current_equity > self.today_stats.get('peak_equity', 0.0):
            self.today_stats['peak_equity'] = current_equity

        # Update maximum daily drawdown
        peak = self.today_stats.get('peak_equity', 0.0)
        if peak > 0:
            drawdown_pct = ((peak - current_equity) / peak) * 100
            if drawdown_pct > self.today_stats.get('max_daily_drawdown_pct', 0.0):
                self.today_stats['max_daily_drawdown_pct'] = drawdown_pct
                self.today_stats['max_daily_drawdown_dollars'] = peak - current_equity

        # Update the win rate for the day
        self.today_stats['end_of_day_win_rate_pct'] = win_rate

        self.logger.info(
            f"Daily Stats: Peak Equity: {self.today_stats.get('peak_equity', 0.0):.2f}, "
            f"Max Daily Drawdown: {self.today_stats.get('max_daily_drawdown_pct', 0.0):.2f}% "
            f"(${self.today_stats.get('max_daily_drawdown_dollars', 0.0):.2f})"
        )
        self._save_stats()  # Save the updated stats

    def update_daily_profit(self, daily_profit):
        """
        Updates the daily profit and saves the stats.
        Called from: TradingBot.update_and_calculate_win_rate
        """
        self._check_date_rollover(None)  # Check for date change
        self.today_stats['daily_profit'] = daily_profit  # Update the daily profit value
        self.logger.info(f"Daily Stats: Daily Profit: {self.today_stats.get('daily_profit', 0.0):.2f}")
        self._save_stats()  # Save the updated stats

    def update_trade_stats(self, trades_opened, trades_closed, sl_hits, tp_hits):
        """
        Updates the daily trade statistics in the performance file.
        Called from: TradingBot.update_and_calculate_win_rate
        """
        self._check_date_rollover(None)  # Check for date change
        # Update the trade counts for the day
        self.today_stats['trades_opened'] = trades_opened
        self.today_stats['trades_closed'] = trades_closed
        self.today_stats['sl_hits'] = sl_hits
        self.today_stats['tp_hits'] = tp_hits
        # Calculate hit rates as percentages
        if trades_closed > 0:
            self.today_stats['sl_hit_rate_pct'] = (sl_hits / trades_closed) * 100
            self.today_stats['tp_hit_rate_pct'] = (tp_hits / trades_closed) * 100
        else:
            self.today_stats['sl_hit_rate_pct'] = 0.0
            self.today_stats['tp_hit_rate_pct'] = 0.0
        self._save_stats()  # Save the updated stats


# --- Main Trading Bot Class ---
class TradingBot:
    """The main class for the trading bot logic."""

    def __init__(self, user, password, server, path, indices, max_open_trades_per_asset, trade_marker_size):
        """
        Initializes the TradingBot instance.
        This function is the constructor for the bot and sets up all initial parameters.
        Called from: The main execution block at the end of the file.
        """
        self.logger = logger  # Assign the logger instance
        self.user = user  # MT5 username
        self.password = password  # MT5 password
        self.server = server  # MT5 server
        self.path = path  # Path to MT5 terminal
        self.indices = indices  # List of symbols to trade
        self.max_open_trades_per_asset = max_open_trades_per_asset # Max open trades per asset
        self.trade_marker_size = trade_marker_size # Size of the buy/sell markers on the plot

        # Dictionary to store historical candle data for each symbol and timeframe
        self.data_history = {tf: {} for tf in [TF_TREND, TF_ENTRY]}
        # Dictionary to store the current trend for each symbol
        self.trends = {symbol: 'NONE' for symbol in self.indices}

        # Setup for the trade journal CSV file
        self.trade_journal_file = os.path.join(OUTPUT_DIR, 'trading_journal_mt5.csv')
        self.journal_header = [
            'timestamp', 'symbol', 'trade_type', 'entry_price', 'exit_price',
            'pnl', 'outcome', 'notes', 'sl', 'tp', 'ticket'
        ]
        self.init_journal()  # Initialize the journal file

        # Initialize performance and drawdown tracking variables
        self.current_equity = 0
        self.current_peak = 0.0
        self.overall_max_drawdown = 0.0
        self.overall_max_drawdown_dollars = 0.0
        self.win_rate = 0.0

        # Initialize trade counters (total and daily)
        self.trades_opened_total = 0
        self.trades_closed_total = 0
        self.sl_hits_total = 0
        self.tp_hits_total = 0
        self.trades_opened_daily = 0
        self.trades_closed_daily = 0
        self.sl_hits_daily = 0
        self.tp_hits_daily = 0

        self.mt5_connected = False  # Flag to track MT5 connection status
        # Create an instance of the daily performance tracker
        self.performance_tracker = DailyPerformanceTracker(self.logger)
        self._initialize_counters_from_journal() # Initialize counters from the journal

    def _initialize_counters_from_journal(self):
        """
        Initializes trade counters by reading the trade journal.
        This ensures that the bot's state is consistent with the journal upon startup.
        Called from: __init__
        """
        try:
            if os.path.exists(self.trade_journal_file):
                journal_df = pd.read_csv(self.trade_journal_file)
                if not journal_df.empty:
                    # Initialize total counters from the entire journal
                    self.trades_opened_total = len(journal_df)
                    closed_trades = journal_df[journal_df['notes'] != 'OPEN']
                    self.trades_closed_total = len(closed_trades)
                    self.sl_hits_total = len(closed_trades[closed_trades['outcome'] == 'LOSS'])
                    self.tp_hits_total = len(closed_trades[closed_trades['outcome'] == 'WIN'])
                    self.logger.info(f"Initialized TOTAL counters from journal: Opened={self.trades_opened_total}, Closed={self.trades_closed_total}, SL={self.sl_hits_total}, TP={self.tp_hits_total}")

                    # Initialize daily counters for the current day
                    today_str = datetime.now(pytz.utc).strftime('%Y-%m-%d')
                    journal_df['timestamp'] = pd.to_datetime(journal_df['timestamp'])
                    
                    # Trades opened today
                    daily_opened_mask = journal_df['timestamp'].dt.strftime('%Y-%m-%d') == today_str
                    self.trades_opened_daily = daily_opened_mask.sum()

                    # Trades closed today
                    daily_closed_mask = journal_df['notes'].str.startswith(f'Closed at {today_str}', na=False)
                    daily_closed_trades = journal_df[daily_closed_mask]
                    self.trades_closed_daily = len(daily_closed_trades)
                    self.sl_hits_daily = len(daily_closed_trades[daily_closed_trades['outcome'] == 'LOSS'])
                    self.tp_hits_daily = len(daily_closed_trades[daily_closed_trades['outcome'] == 'WIN'])
                    self.logger.info(f"Initialized DAILY counters for {today_str}: Opened={self.trades_opened_daily}, Closed={self.trades_closed_daily}, SL={self.sl_hits_daily}, TP={self.tp_hits_daily}")

        except Exception as e:
            self.logger.error(f"Error initializing counters from journal: {e}")

    def _reset_daily_counters(self):
        """
        Resets the daily trade counters to zero.
        Called from: DailyPerformanceTracker._check_date_rollover
        """
        self.logger.info("Resetting daily trade counters.")
        self.trades_opened_daily = 0
        self.trades_closed_daily = 0
        self.sl_hits_daily = 0
        self.tp_hits_daily = 0

    def print_trade_statistics(self):
        """
        Prints the current trade statistics to the log.
        Called from: run_bot_iteration
        """
        self.logger.info("--- Trade Statistics (Today) ---")
        
        # Get current open trades from MT5, filtering by magic number
        open_positions = mt5.positions_get()
        if open_positions is None:
            open_positions = []
        
        bot_positions = [p for p in open_positions if p.magic == 20250824]
        open_trades_count = len(bot_positions)
        self.logger.info(f"Current Open Trades: {open_trades_count}")

        self.logger.info(f"Total Closed Trades Today: {self.trades_closed_daily}")
        
        # Calculate SL and TP hit rates for today, handling division by zero
        sl_percentage = (self.sl_hits_daily / self.trades_closed_daily * 100) if self.trades_closed_daily > 0 else 0
        tp_percentage = (self.tp_hits_daily / self.trades_closed_daily * 100) if self.trades_closed_daily > 0 else 0
        
        self.logger.info(f"Stop Loss Hits Today: {self.sl_hits_daily} ({sl_percentage:.2f}% of closed)")
        self.logger.info(f"Take Profit Hits Today: {self.tp_hits_daily} ({tp_percentage:.2f}% of closed)")
        
        # Calculate and print Daily Win Rate based on MT5 data
        daily_win_rate = (self.tp_hits_daily / self.trades_closed_daily * 100) if self.trades_closed_daily > 0 else 0
        self.logger.info(f"Daily Win Rate: {daily_win_rate:.2f}%")

        # Print Overall Win Rate (from journal)
        self.logger.info(f"Overall Win Rate (Journal): {self.win_rate:.2f}%")
        self.logger.info("---------------------------------")

    def init_journal(self):
        """
        Creates or validates the trade journal file and its header.
        """
        try:
            file_exists = os.path.exists(self.trade_journal_file)
            
            if not file_exists:
                # If file doesn't exist, create it with the correct header
                with open(self.trade_journal_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.journal_header)
                self.logger.info(f"Created new journal file: {self.trade_journal_file}")
            else:
                # If file exists, check if the header is correct
                with open(self.trade_journal_file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    try:
                        header = next(reader)
                        if header != self.journal_header:
                            self.logger.warning("Journal header is incorrect. Rewriting file with correct header.")
                            # Read all data
                            data = list(reader)
                            # Rewrite the whole file
                            with open(self.trade_journal_file, 'w', newline='') as wf:
                                writer = csv.writer(wf)
                                writer.writerow(self.journal_header)
                                writer.writerows(data)
                    except StopIteration:
                        # File is empty, write header
                        with open(self.trade_journal_file, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(self.journal_header)
                        self.logger.info(f"Journal file was empty. Wrote header.")

        except Exception as e:
            self.logger.error(f"Error initializing or validating journal file: {e}", exc_info=True)

    def connect_mt5(self, max_retries=3, delay_seconds=5):
        """
        Attempts to connect to MetaTrader 5 with a retry mechanism.
        Called from: run_bot_iteration, plot_symbol_data
        """
        if self.mt5_connected:  # If already connected, do nothing
            return True
        for attempt in range(max_retries):  # Loop for a specified number of retries
            try:
                # Attempt to initialize the MT5 connection
                if not mt5.initialize(login=self.user, server=self.server,
                                    password=self.password, path=self.path):
                    self.logger.error(f"MT5 initialize() failed, error: {mt5.last_error()}")
                    time.sleep(delay_seconds)  # Wait before retrying
                    continue
                # Attempt to log in to the account
                if not mt5.login(login=self.user, server=self.server,
                                password=self.password):
                    self.logger.error(f"MT5 login() failed, error: {mt5.last_error()}")
                    mt5.shutdown()  # Shutdown the connection on login failure
                    time.sleep(delay_seconds)  # Wait before retrying
                    continue
                self.mt5_connected = True  # Set connection flag to True on success
                self.logger.info("Connected to MetaTrader 5 successfully.")
                return True  # Exit the function on successful connection
            except Exception as e:
                self.logger.critical(f"Error connecting to MT5 on attempt {attempt + 1}: {e}", exc_info=True)
                time.sleep(delay_seconds)  # Wait before retrying
        self.logger.critical("Failed to connect to MT5 after multiple retries.")
        self.mt5_connected = False  # Set connection flag to False on failure
        return False

    def disconnect_mt5(self):
        """
        Disconnects from the MetaTrader 5 terminal.
        Called from: plot_symbol_data, main execution block (on exit)
        """
        if self.mt5_connected:  # Only disconnect if connected
            mt5.shutdown()  # Call the MT5 shutdown function
            self.mt5_connected = False  # Update the connection flag
            self.logger.info("Disconnected from MetaTrader 5.")

    def get_candle_data(self, symbol, timeframe, count=500):
        """
        Fetches historical candle (rate) data for a specific symbol and timeframe.
        Called from: run_bot_iteration, plot_symbol_data
        """
        if not self.mt5_connected:  # Check for MT5 connection
            return pd.DataFrame()  # Return an empty DataFrame if not connected
        try:
            # Ensure the symbol is available in the Market Watch
            if not mt5.symbol_select(symbol, True):
                self.logger.warning(f"Failed to select symbol {symbol}, error: {mt5.last_error()}")
                return pd.DataFrame()
            # Request historical data from the current position
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                self.logger.warning(f"No rates for {symbol} {timeframe}, error: {mt5.last_error()}")
                return pd.DataFrame()
            # Convert the received tuple of rates into a pandas DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)  # Convert timestamp to datetime objects
            df = df.set_index('time')  # Set the 'time' column as the DataFrame index
            return df[['open', 'high', 'low', 'close', 'tick_volume']]  # Return the relevant columns
        except Exception as e:
            self.logger.error(f"Exception in get_candle_data for {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def update_data_history(self, symbol, timeframe, new_df):
        """
        Updates the historical data dictionary with the latest candle data.
        Called from: run_bot_iteration
        """
        if new_df.empty:  # Do nothing if the new data is empty
            return
        if symbol not in self.data_history[timeframe]:
            # If it's the first time getting data for this symbol/timeframe, just assign it
            self.data_history[timeframe][symbol] = new_df
        else:
            # If data already exists, combine the old and new data
            combined_df = pd.concat([self.data_history[timeframe][symbol], new_df])
            # Remove any duplicate entries, keeping the last one
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df.sort_index(inplace=True)  # Sort the data by time
            # Keep only the most recent 500 candles to manage memory
            self.data_history[timeframe][symbol] = combined_df.tail(500)

    def update_symbol_trend(self, symbol):
        """
        Analyzes the higher timeframe (TF_TREND) to determine the dominant trend.
        Called from: run_bot_iteration
        """
        df = self.data_history[TF_TREND].get(symbol)  # Get the trend timeframe data
        if df is None or len(df) < EMA_TREND_PERIOD:  # Check if there's enough data
            return
        # Rename columns to be compatible with pandas_ta
        df_ta = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
        # Calculate the trend EMA
        df_ta['ema_trend'] = ta.ema(df_ta.Close, length=EMA_TREND_PERIOD)
        # Calculate MACD indicator
        macd_data = ta.macd(df_ta.Close, fast=12, slow=26, signal=9)
        if macd_data.empty:
            self.trends[symbol] = 'NONE'
            return
        df_ta = pd.concat([df_ta, macd_data], axis=1)  # Combine MACD data with the DataFrame

        # Add a check here
        if df_ta.empty or any(col not in df_ta.columns for col in ['Close', 'ema_trend', 'MACD_12_26_9', 'MACDs_12_26_9']):
             self.logger.warning(f"Trend analysis for {symbol} skipped due to missing data columns after indicator calculation.")
             self.trends[symbol] = 'NONE'
             return

        # Get the latest values
        current_close = df_ta.Close.iloc[-1]
        current_ema_trend = df_ta.ema_trend.iloc[-1]
        current_macd = df_ta['MACD_12_26_9'].iloc[-1]
        current_macds = df_ta['MACDs_12_26_9'].iloc[-1]
        new_trend = 'NONE'  # Default trend is NONE
        # Determine trend based on price relative to EMA and MACD crossover
        if current_close > current_ema_trend and current_macd > current_macds:
            new_trend = 'UP'
        elif current_close < current_ema_trend and current_macd < current_macds:
            new_trend = 'DOWN'
        # Log if the trend has changed
        if self.trends[symbol] != new_trend:
            self.logger.info(f"Trend for {symbol} changed from {self.trends[symbol]} to {new_trend}")
            self.trends[symbol] = new_trend  # Update the trend

    def check_entry_signal(self, symbol):
        """
        Analyzes the lower timeframe (TF_ENTRY) for a pullback entry signal.
        Called from: run_bot_iteration
        """
        trend = self.trends.get(symbol, 'NONE')  # Get the current trend for the symbol
        if trend == 'NONE':  # If there is no clear trend, no signal
            return None
        df = self.data_history[TF_ENTRY].get(symbol)  # Get the entry timeframe data
        if df is None or len(df) < EMA_ENTRY_PERIOD or len(df) < 2:  # Check if there's enough data
            return None
        # Rename columns for pandas_ta
        df_ta = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
        # Calculate the entry EMA
        df_ta['ema_entry'] = ta.ema(df_ta.Close, length=EMA_ENTRY_PERIOD)
        # Get the last two close prices and EMA values
        current_close = df_ta.Close.iloc[-1]
        prev_close = df_ta.Close.iloc[-2]
        current_ema_entry = df_ta.ema_entry.iloc[-1]
        prev_ema_entry = df_ta.ema_entry.iloc[-2]
        # Check for a buy signal (in an uptrend, price crosses above the entry EMA)
        if (trend == 'UP' and current_close > current_ema_entry and
                prev_close <= prev_ema_entry):
            self.logger.info(f"BUY signal for {symbol}: Price crossed above {EMA_ENTRY_PERIOD} EMA on M1 in an UP trend.")
            return 'buy'
        # Check for a sell signal (in a downtrend, price crosses below the entry EMA)
        if (trend == 'DOWN' and current_close < current_ema_entry and
                prev_close >= prev_ema_entry):
            self.logger.info(f"SELL signal for {symbol}: Price crossed below {EMA_ENTRY_PERIOD} EMA on M1 in a DOWN trend.")
            return 'sell'
        return None  # No signal found

    def place_order(self, symbol, trade_type, volume, price, sl, tp):
        """
        Places a market order on MetaTrader 5.
        Called from: manage_trade
        """
        if not self.mt5_connected:
            self.logger.error("MT5 not connected. Cannot place order.")
            return None
        # Create the request dictionary for the order
        request = {
            'action': mt5.TRADE_ACTION_DEAL,  # Action type for market orders
            'symbol': symbol,
            'volume': volume,
            'type': (mt5.ORDER_TYPE_BUY if trade_type == 'buy' else mt5.ORDER_TYPE_SELL),
            'price': price,
            'sl': sl,
            'tp': tp,
            'deviation': 10,  # Allowed price deviation
            'magic': 20250824,  # Magic number to identify trades from this bot
            'comment': "GeminiBot M5/M1 EMA",
            'type_time': mt5.ORDER_TIME_GTC,  # Good 'til Canceled
            'type_filling': mt5.ORDER_FILLING_FOK,  # Fill Or Kill filling policy
        }
        result = mt5.order_send(request)  # Send the order request to the MT5 server
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            # Log an error if the order failed
            self.logger.error(f"Order failed for {symbol}. Error: {result.comment} (retcode: {result.retcode})")
            return None
        else:
            # Log success and return the deal ticket ID
            self.logger.info(f"Order placed successfully for {symbol}. Deal ID: {result.deal}")
            return result.deal

    def _deals_to_dataframe(self, deals):
        """Converts a list of MT5 deals to a pandas DataFrame."""
        if deals is None or not deals:
            return pd.DataFrame()
        try:
            # The result of history_deals_get is a tuple of Deal objects
            df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            return df
        except (AttributeError, IndexError) as e:
            self.logger.error(f"Could not convert deals to DataFrame: {e}")
            return pd.DataFrame()

    def _orders_to_dataframe(self, orders):
        """Converts a list of MT5 orders to a pandas DataFrame."""
        if orders is None or not orders:
            return pd.DataFrame()
        try:
            df = pd.DataFrame(list(orders), columns=orders[0]._asdict().keys())
            df['time_setup'] = pd.to_datetime(df['time_setup'], unit='s', utc=True)
            df['time_done'] = pd.to_datetime(df['time_done'], unit='s', utc=True)
            return df
        except (AttributeError, IndexError) as e:
            self.logger.error(f"Could not convert orders to DataFrame: {e}")
            return pd.DataFrame()

    def manage_trade(self, symbol, signal):
        """
        Calculates trade parameters (volume, SL, TP) and places the order.
        Called from: run_bot_iteration
        """
        # Check if the number of open positions for this symbol exceeds the limit
        open_positions = mt5.positions_get(symbol=symbol)

        # Get custom max open trades for the symbol, or use the global default
        max_trades_for_asset = CUSTOM_MAX_OPEN_TRADES.get(symbol, self.max_open_trades_per_asset)

        if open_positions and len(open_positions) >= max_trades_for_asset:
            self.logger.info(
                f"Skipping {signal} signal for {symbol} as the maximum number of open "
                f"trades ({max_trades_for_asset}) has been reached."
            )
            return

        # Get symbol information (e.g., digits, volume limits)
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Could not get symbol info for {symbol}.")
            return

        # Get the latest tick data (bid/ask prices)
        tick_info = mt5.symbol_info_tick(symbol)
        if tick_info is None:
            self.logger.error(f"Could not get tick info for {symbol}.")
            return

        # Determine the entry price based on the signal type
        current_price = tick_info.ask if signal == 'buy' else tick_info.bid

        # Get recent data for ATR calculation
        df_for_atr = self.data_history[TF_ENTRY][symbol].tail(15)
        if len(df_for_atr) < 14:
            self.logger.warning(f"Not enough data for ATR calc for {symbol}. Skipping order.")
            return

        # Calculate ATR (Average True Range) for setting SL
        df_ta = df_for_atr.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
        atr_val = ta.atr(df_ta.High, df_ta.Low, df_ta.Close, length=14).iloc[-1]
        if pd.isna(atr_val) or atr_val == 0:
            self.logger.warning(f"ATR calc resulted in invalid value ({atr_val}) for {symbol}.")
            return

        # Determine the trade volume, respecting symbol limits and custom settings
        volume = (CUSTOM_LOT_SIZES.get(symbol) or symbol_info.volume_min)
        volume = max(volume, symbol_info.volume_min)
        volume = min(volume, symbol_info.volume_max)
        volume = round(volume / symbol_info.volume_step) * symbol_info.volume_step
        volume = round(volume, 8)

        # Get custom risk parameters for the symbol, or use the global defaults
        risk_params = CUSTOM_RISK_PARAMETERS.get(symbol, {})
        sl_multiplier = risk_params.get('sl_multiplier', SL_MULTIPLIER)
        tp_risk_reward = risk_params.get('tp_risk_reward', TP_RISK_REWARD)
        self.logger.info(f"Using risk params for {symbol}: SL Mult={sl_multiplier}, TP R:R={tp_risk_reward}")

        # Calculate SL and TP levels
        digits = symbol_info.digits
        spread = tick_info.ask - tick_info.bid
        sl_points = sl_multiplier * atr_val
        if signal == 'buy':
            sl = current_price - sl_points
            tp = current_price + (tp_risk_reward * sl_points) + spread
        else:  # sell
            sl = current_price + sl_points
            tp = current_price - (tp_risk_reward * sl_points) - spread

        # Adjust SL and TP to meet broker requirements (e.g., minimum distance from price)
        sl, tp = self._adjust_stops_for_broker_rules(symbol_info, tick_info, signal, sl, tp, current_price)

        if sl is None or tp is None:
            self.logger.error(f"Failed to determine valid SL/TP for {symbol}. Skipping trade.")
            return

        self.logger.info(
            f"Placing Order: {signal} for {symbol} @ {current_price:.{digits}f}, "
            f"SL: {sl:.{digits}f}, TP: {tp:.{digits}f}, Vol: {volume}"
        )
        # Place the actual order
        deal_ticket = self.place_order(symbol, signal, volume, current_price, sl, tp)
        if deal_ticket:
            # If the order is placed successfully, update counters and journal the trade
            self.trades_opened_total += 1
            self.trades_opened_daily += 1
            self.journal_trade(symbol, signal, current_price, sl, tp, deal_ticket)

    def _adjust_stops_for_broker_rules(self, symbol_info, tick_info, signal, sl, tp, entry_price):
        """
        Adjusts SL/TP to meet the broker's minimum distance requirement (stops level)
        and ensures the risk-reward ratio is at least 1:1.
        """
        digits = symbol_info.digits
        min_stop_dist = symbol_info.trade_stops_level * symbol_info.point

        if signal == 'buy':
            # Adjust SL if too close to current bid
            if (tick_info.bid - sl) < min_stop_dist:
                original_sl = sl
                sl = tick_info.bid - min_stop_dist
                self.logger.warning(
                    f"Original SL {original_sl:.{digits}f} for {symbol_info.name} was too close. "
                    f"Adjusting SL to {sl:.{digits}f}."
                )

            # Adjust TP if too close to current ask
            if (tp - tick_info.ask) < min_stop_dist:
                original_tp = tp
                tp = tick_info.ask + min_stop_dist
                self.logger.warning(
                    f"Original TP {original_tp:.{digits}f} for {symbol_info.name} was too close. "
                    f"Adjusting TP to {tp:.{digits}f}."
                )
            
            risk = entry_price - sl
            reward = tp - entry_price
            
            if risk <= 0:
                self.logger.error(f"Risk is zero or negative for {symbol_info.name} BUY after adjustment. Cannot place trade.")
                return None, None

            # Enforce minimum 1:1 R:R
            if reward < risk:
                new_tp = entry_price + risk
                self.logger.warning(
                    f"R:R for {symbol_info.name} was less than 1:1. "
                    f"Adjusting TP from {tp:.{digits}f} to {new_tp:.{digits}f} to enforce 1:1."
                )
                tp = new_tp

        else:  # sell
            # Adjust SL if too close to current ask
            if (sl - tick_info.ask) < min_stop_dist:
                original_sl = sl
                sl = tick_info.ask + min_stop_dist
                self.logger.warning(
                    f"Original SL {original_sl:.{digits}f} for {symbol_info.name} was too close. "
                    f"Adjusting SL to {sl:.{digits}f}."
                )

            # Adjust TP if too close to current bid
            if (tick_info.bid - tp) < min_stop_dist:
                original_tp = tp
                tp = tick_info.bid - min_stop_dist
                self.logger.warning(
                    f"Original TP {original_tp:.{digits}f} for {symbol_info.name} was too close. "
                    f"Adjusting TP to {tp:.{digits}f}."
                )

            risk = sl - entry_price
            reward = entry_price - tp

            if risk <= 0:
                self.logger.error(f"Risk is zero or negative for {symbol_info.name} SELL after adjustment. Cannot place trade.")
                return None, None

            # Enforce minimum 1:1 R:R
            if reward < risk:
                new_tp = entry_price - risk
                self.logger.warning(
                    f"R:R for {symbol_info.name} was less than 1:1. "
                    f"Adjusting TP from {tp:.{digits}f} to {new_tp:.{digits}f} to enforce 1:1."
                )
                tp = new_tp

        return round(sl, digits), round(tp, digits)

    def journal_trade(self, symbol, trade_type, entry_price, sl, tp, ticket):
        """
        Writes the initial details of a newly opened trade to the journal CSV file.
        Called from: manage_trade
        """
        timestamp = datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')  # Get current UTC time
        # Prepare the data for the new row in the journal
        trade_data = [
            timestamp, symbol, trade_type, entry_price, '', '', '', 'OPEN',
            sl, tp, ticket
        ]
        try:
            # Open the journal file in append mode to add the new trade
            with open(self.trade_journal_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(trade_data)
        except Exception as e:
            self.logger.error(f"Error writing to trade journal: {e}", exc_info=True)

    def update_and_calculate_win_rate(self):
        """
        Checks for closed trades, updates the journal, and calculates win rate and other stats.
        Called from: run_bot_iteration
        """
        if not self.mt5_connected:
            return
        try:
            # Read the trade journal, ensuring the ticket is treated as an integer that can be null
            journal_df = pd.read_csv(
                self.trade_journal_file, dtype={'ticket': 'Int64'}
            )
        except FileNotFoundError:
            return  # If journal doesn't exist yet
        except Exception as e:
            self.logger.error(f"Error reading journal file: {e}")
            return

        # Filter for trades that are still marked as 'OPEN'
        open_trades_df = journal_df[journal_df['notes'] == 'OPEN'].copy()
        if open_trades_df.empty:
            # Even if no open trades, we still want to update daily stats from MT5
            self.logger.info("No open trades found in journal.")

        # Get all deals from the last 7 days to check against
        from_date = datetime.now(pytz.utc) - timedelta(days=7)
        all_deals = mt5.history_deals_get(from_date, datetime.now(pytz.utc))
        deals_df = self._deals_to_dataframe(all_deals)

        if deals_df.empty:
            self.logger.info("No historical deals found in the last 7 days.")
        else:
            deals_df = deals_df[deals_df['magic'] == 20250824]
            if deals_df.empty:
                self.logger.info(f"No deals with magic number 20250824 found in the last 7 days.")
            else:
                self.logger.info(f"Found {len(deals_df)} deals with magic number 20250824 in the last 7 days.")

        # --- Get today's closed trades directly from MT5 ---
        today_start = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = datetime.now(pytz.utc)

        all_deals_today = mt5.history_deals_get(today_start, today_end)
        deals_df_today = self._deals_to_dataframe(all_deals_today)
        
        current_day_closed_trades = 0
        current_day_sl_hits = 0
        current_day_tp_hits = 0

        if not deals_df_today.empty:
            deals_df_today = deals_df_today[deals_df_today['magic'] == 20250824] # Filter by bot's magic number
            self.logger.info(f"DEBUG: Deals after magic number filter: {len(deals_df_today)} deals.")

            # Filter for closing deals (DEAL_ENTRY_OUT)
            closed_deals_today = deals_df_today[deals_df_today['entry'] == mt5.DEAL_ENTRY_OUT]
            self.logger.info(f"DEBUG: Closed deals after entry type filter: {len(closed_deals_today)} deals.")
            
            current_day_closed_trades = len(closed_deals_today)
            current_day_sl_hits = len(closed_deals_today[closed_deals_today['profit'] <= 0])
            current_day_tp_hits = len(closed_deals_today[closed_deals_today['profit'] > 0])

        # Update the daily counters with values from MT5
        self.trades_closed_daily = current_day_closed_trades
        self.sl_hits_daily = current_day_sl_hits
        self.tp_hits_daily = current_day_tp_hits
        self.logger.info(f"Daily closed trades from MT5: {self.trades_closed_daily}, SL: {self.sl_hits_daily}, TP: {self.tp_hits_daily}")

        # --- Continue with updating journal for trades that were closed ---
        trades_updated = False
        # Iterate through each open trade in the journal
        for index, trade in open_trades_df.iterrows():
            ticket = trade['ticket']
            if pd.isna(ticket):
                continue
            try:
                # Find the deal that opened this position
                opening_deal = deals_df[deals_df['ticket'] == ticket]
                if opening_deal.empty:
                    self.logger.warning(f"Could not find opening deal for ticket {ticket} in recent history.")
                    continue
                pos_id = opening_deal.iloc[0]['position_id']
                self.logger.info(f"Checking status for ticket {ticket}, position_id {pos_id}.")
                
                # Find any deals that closed this position
                closing_deals = deals_df[
                    (deals_df['position_id'] == pos_id) &
                    (deals_df['entry'] == mt5.DEAL_ENTRY_OUT)
                ]
                if not closing_deals.empty:
                    self.logger.info(f"Found {len(closing_deals)} closing deal(s) for position {pos_id}.")
                    # If closing deals are found, the trade is closed
                    # Calculate total profit including swaps and commissions
                    profit = closing_deals['profit'].sum() + closing_deals['swap'].sum() + closing_deals['commission'].sum()
                    exit_price = closing_deals.iloc[-1]['price']
                    exit_time = pd.to_datetime(
                        closing_deals.iloc[-1]['time'], unit='s', utc=True
                    )
                    outcome = 'WIN' if profit > 0 else 'LOSS'

                    # Update the journal DataFrame with the closing information
                    journal_df.loc[index, 'exit_price'] = exit_price
                    journal_df.loc[index, 'pnl'] = profit
                    journal_df.loc[index, 'outcome'] = outcome
                    journal_df.loc[index, 'notes'] = f"Closed at {exit_time:%Y-%m-%d %H:%M}"
                    trades_updated = True
                    self.logger.info(
                        f"Trade Closed: {trade['symbol']} (Ticket: {ticket}), "
                        f"PnL: {profit:.2f}, Outcome: {outcome}"
                    )
            except Exception as e:
                self.logger.error(f"Error processing status for ticket {trade.get('ticket')}: {e}", exc_info=True)

        if trades_updated:
            # If any trades were updated, save the journal back to the CSV file
            journal_df.to_csv(self.trade_journal_file, index=False)
            self.logger.info("Trade journal has been updated.")
        else:
            self.logger.info("Checked open trades. None were found to be closed.")

        try:
            # Calculate and update daily profit
            today_str = datetime.now(pytz.utc).strftime('%Y-%m-%d')

            # This part still uses journal_df for daily profit, which is fine.
            todays_closed_trades_for_profit = journal_df[
                journal_df['notes'].str.startswith(f'Closed at {today_str}', na=False)
            ]

            daily_profit = todays_closed_trades_for_profit['pnl'].sum()
            self.performance_tracker.update_daily_profit(daily_profit)
            self.logger.info(f"Updated daily profit for {today_str}: {daily_profit:.2f}")

        except Exception as e:
            self.logger.error(f"Error calculating daily profit: {e}", exc_info=True)

        # Calculate and log the overall win rate
        closed = journal_df[journal_df['outcome'].isin(['WIN', 'LOSS'])]
        if not closed.empty:
            wins = len(closed[closed['outcome'] == 'WIN'])
            total = len(closed)
            self.win_rate = (wins / total) * 100 if total > 0 else 0
            self.logger.info(f"Overall Win Rate: {self.win_rate:.2f}% ({wins}/{total})")
        
        # Update the daily performance CSV with the latest trade stats for the day
        # These values are now directly from MT5 query at the beginning of the function
        self.performance_tracker.update_trade_stats(
            self.trades_opened_daily,
            self.trades_closed_daily,
            self.sl_hits_daily,
            self.tp_hits_daily
        )

    def update_drawdown_calculations(self):
        """
        Updates equity and calculates overall and daily drawdown.
        Called from: run_bot_iteration
        """
        if not self.mt5_connected:
            return
        try:
            account_info = mt5.account_info()  # Get current account information
            if account_info is None:
                self.logger.warning("Could not retrieve account info.")
                return
            self.current_equity = account_info.equity  # Update current equity
            # Set the initial peak equity if it hasn't been set yet
            if self.current_peak == 0.0:
                self.current_peak = self.current_equity
                self.logger.info(f"Initial equity peak set to: {self.current_peak:.2f}")
            # Update the peak equity if a new high is reached
            if self.current_equity > self.current_peak:
                self.current_peak = self.current_equity
            # Calculate and update the maximum overall drawdown
            if self.current_peak > 0:
                drawdown = (self.current_peak - self.current_equity) / self.current_peak
                if drawdown > self.overall_max_drawdown:
                    self.overall_max_drawdown = drawdown
                    self.overall_max_drawdown_dollars = self.current_peak - self.current_equity
                    self.logger.info(f"New max overall drawdown: {self.overall_max_drawdown:.2%} (${self.overall_max_drawdown_dollars:.2f})")
            self.logger.info(
                f"Overall Stats: Equity: {self.current_equity:.2f}, "
                f"Peak: {self.current_peak:.2f}, "
                f"Max Drawdown: {self.overall_max_drawdown:.2%} (${self.overall_max_drawdown_dollars:.2f})"
            )
        except Exception as e:
            self.logger.error(f"Error during drawdown calculation: {e}", exc_info=True)

    def run_bot_iteration(self):
        """
        The main loop of the bot, executed periodically by the scheduler.
        Called from: main execution block (via scheduler)
        """
        if not self.connect_mt5():  # Ensure connection to MT5
            self.logger.error("Halting iteration due to connection failure.")
            return
        self.logger.info(f"--- Running bot iteration at {datetime.now(pytz.utc):%Y-%m-%d %H:%M:%S} UTC ---")
        # Perform all per-iteration tasks
        self.update_drawdown_calculations()
        self.update_and_calculate_win_rate()
        self.performance_tracker.update(self.current_equity, self.win_rate, self)
        self.print_trade_statistics()
        self.logger.info("--- Scanning for new trade opportunities ---")
        # Loop through each symbol to check for trading signals
        for symbol in self.indices:
            try:
                # Get the latest candle data for both timeframes
                df_trend = self.get_candle_data(
                    symbol, TF_TREND, count=EMA_TREND_PERIOD + 50
                )
                df_entry = self.get_candle_data(
                    symbol, TF_ENTRY, count=EMA_ENTRY_PERIOD + 50
                )
                # Update the historical data
                self.update_data_history(symbol, TF_TREND, df_trend)
                self.update_data_history(symbol, TF_ENTRY, df_entry)
                # Determine the trend and check for entry signals
                self.update_symbol_trend(symbol)
                signal = self.check_entry_signal(symbol)
                if signal:
                    # If a signal is found, manage and place the trade
                    self.manage_trade(symbol, signal)
            except Exception as e:
                self.logger.critical(f"Unhandled error in bot iteration for {symbol}: {e}", exc_info=True)

    def run_backtest(self, days=30):
        """
        Runs a backtest of the strategy over historical data.
        """
        self.logger.info(f"--- Starting Backtest Mode for the last {days} days ---")
        if not self.connect_mt5():
            self.logger.error("Cannot run backtest: MT5 connection failed.")
            return

        overall_results = []

        for symbol in self.indices:
            self.logger.info(f"--- Backtesting Symbol: {symbol} ---")

            # 1. Fetch Data
            num_candles = days * 24 * 60  # days * hours * minutes
            self.logger.info(f"Fetching {num_candles} M1 candles for backtesting...")
            df_m1 = self.get_candle_data(symbol, TF_ENTRY, count=num_candles)
            if df_m1 is None or df_m1.empty or len(df_m1) < EMA_TREND_PERIOD:
                self.logger.warning(f"Not enough M1 data for {symbol} to run backtest. Skipping.")
                continue
            
            self.logger.info(f"Fetching {days * 24 * 12} M5 candles for backtesting...")
            df_m5 = self.get_candle_data(symbol, TF_TREND, count=days * 24 * 12) # Approx num M5 candles in a period
            if df_m5 is None or df_m5.empty or len(df_m5) < EMA_TREND_PERIOD:
                self.logger.warning(f"Not enough M5 data for {symbol} to run backtest. Skipping.")
                continue

            # 2. Simulate
            simulated_trades = []
            open_trade = None

            for i in range(EMA_TREND_PERIOD, len(df_m1)):
                # Create historical slices for the current point in time
                current_time = df_m1.index[i]
                history_m1 = df_m1.iloc[:i+1]
                history_m5 = df_m5[df_m5.index <= current_time]

                if history_m5.empty:
                    continue

                # Update data history for strategy methods
                self.data_history[TF_ENTRY][symbol] = history_m1
                self.data_history[TF_TREND][symbol] = history_m5

                # Check for exit on the open trade first
                if open_trade:
                    candle = df_m1.iloc[i]
                    if open_trade['type'] == 'buy':
                        if candle['low'] <= open_trade['sl']:
                            open_trade['outcome'] = 'LOSS'
                            open_trade['exit_price'] = open_trade['sl']
                            open_trade['exit_time'] = candle.name
                            simulated_trades.append(open_trade)
                            open_trade = None
                        elif candle['high'] >= open_trade['tp']:
                            open_trade['outcome'] = 'WIN'
                            open_trade['exit_price'] = open_trade['tp']
                            open_trade['exit_time'] = candle.name
                            simulated_trades.append(open_trade)
                            open_trade = None
                    elif open_trade['type'] == 'sell':
                        if candle['high'] >= open_trade['sl']:
                            open_trade['outcome'] = 'LOSS'
                            open_trade['exit_price'] = open_trade['sl']
                            open_trade['exit_time'] = candle.name
                            simulated_trades.append(open_trade)
                            open_trade = None
                        elif candle['low'] <= open_trade['tp']:
                            open_trade['outcome'] = 'WIN'
                            open_trade['exit_price'] = open_trade['tp']
                            open_trade['exit_time'] = candle.name
                            simulated_trades.append(open_trade)
                            open_trade = None
                
                # If no trade is open, check for a new entry signal
                if not open_trade:
                    self.update_symbol_trend(symbol)
                    signal = self.check_entry_signal(symbol)

                    if signal:
                        # --- Simulate Trade Placement ---
                        entry_price = df_m1['close'].iloc[i]
                        
                        df_for_atr = history_m1.tail(15)
                        if len(df_for_atr) < 14: continue
                        
                        df_ta = df_for_atr.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
                        atr_val = ta.atr(df_ta.High, df_ta.Low, df_ta.Close, length=14).iloc[-1]
                        if pd.isna(atr_val) or atr_val == 0: continue

                        risk_params = CUSTOM_RISK_PARAMETERS.get(symbol, {})
                        sl_multiplier = risk_params.get('sl_multiplier', SL_MULTIPLIER)
                        tp_risk_reward = risk_params.get('tp_risk_reward', TP_RISK_REWARD)
                        
                        sl_points = sl_multiplier * atr_val
                        if signal == 'buy':
                            sl = entry_price - sl_points
                            tp = entry_price + (tp_risk_reward * sl_points)
                        else: # sell
                            sl = entry_price + sl_points
                            tp = entry_price - (tp_risk_reward * sl_points)

                        open_trade = {
                            'symbol': symbol,
                            'type': signal,
                            'entry_price': entry_price,
                            'entry_time': current_time,
                            'sl': sl,
                            'tp': tp,
                            'outcome': 'OPEN'
                        }
                        self.logger.info(f"Backtest: Opened {signal} trade for {symbol} at {entry_price:.5f}")

            # 3. Analyze Results for the symbol
            if not simulated_trades:
                self.logger.warning(f"No trades were simulated for {symbol}. Cannot generate report.")
                continue

            trades_df = pd.DataFrame(simulated_trades)
            wins = trades_df[trades_df['outcome'] == 'WIN']
            losses = trades_df[trades_df['outcome'] == 'LOSS']
            win_rate = (len(wins) / len(trades_df)) * 100 if not trades_df.empty else 0
            
            # SL Analysis
            sl_too_tight_count = 0
            for index, trade in losses.iterrows():
                # Check price action *after* the SL was hit
                future_candles = df_m1[df_m1.index > trade['exit_time']]
                if future_candles.empty: continue

                if trade['type'] == 'buy':
                    # Did price eventually go up to the TP level?
                    reached_tp_later = (future_candles['high'] >= trade['tp']).any()
                    if reached_tp_later:
                        sl_too_tight_count += 1
                elif trade['type'] == 'sell':
                    # Did price eventually go down to the TP level?
                    reached_tp_later = (future_candles['low'] <= trade['tp']).any()
                    if reached_tp_later:
                        sl_too_tight_count += 1

            self.logger.info(f"--- Backtest Report for {symbol} ---")
            self.logger.info(f"Period: Last {days} days")
            self.logger.info(f"Total Trades: {len(trades_df)}")
            self.logger.info(f"Wins: {len(wins)}")
            self.logger.info(f"Losses: {len(losses)}")
            self.logger.info(f"Win Rate: {win_rate:.2f}%")
            self.logger.info("--- SL Analysis ---")
            self.logger.info(f"Trades that hit SL but would have hit TP later: {sl_too_tight_count} ({ (sl_too_tight_count/len(losses))*100 if len(losses)>0 else 0 :.2f}% of losses)")
            self.logger.info("This suggests the SL might be too tight on these occasions.")
            self.logger.info("-------------------------------------")
            overall_results.append(trades_df)

        self.disconnect_mt5()
        self.logger.info("--- Backtest Finished ---")
        if overall_results:
            all_trades_df = pd.concat(overall_results)
            self.logger.info(f"--- Overall Backtest Report ({len(self.indices)} symbols) ---")
            self.logger.info(f"Total Trades: {len(all_trades_df)}")
            wins = all_trades_df[all_trades_df['outcome'] == 'WIN']
            losses = all_trades_df[all_trades_df['outcome'] == 'LOSS']
            win_rate = (len(wins) / len(all_trades_df)) * 100 if not all_trades_df.empty else 0
            self.logger.info(f"Wins: {len(wins)}")
            self.logger.info(f"Losses: {len(losses)}")
            self.logger.info(f"Overall Win Rate: {win_rate:.2f}%")
            self.logger.info("-------------------------------------")


    def plot_symbol_data(self, symbol):
        """
        Fetches required data from MT5, plots the chart with trades, and saves it as a PNG file.
        Called from: main execution block (if mode is 'plot')
        """
        self.logger.info(f"Preparing detailed plot for {symbol} from MT5 data...")

        # --- Step 1: Fetch price data ---
        num_candles = 72 * 60  # 72 hours * 60 minutes/hour
        self.logger.info(f"Fetching latest {num_candles} M1 candles for {symbol} (approx. 72 hours).")
        df_entry = self.get_candle_data(symbol, TF_ENTRY, count=num_candles)
        if df_entry is None or df_entry.empty:
            self.logger.error(f"No M1 data available for {symbol}. Cannot generate plot.")
            return

        # --- Step 2: Fetch Trade History from MT5 ---
        self.logger.info("Fetching trade history from MT5 for the last 72 hours.")
        from_date = datetime.now(pytz.utc) - timedelta(hours=72)
        to_date = datetime.now(pytz.utc)
        deals = mt5.history_deals_get(from_date, to_date)
        
        trades_to_plot = []
        if deals is None or len(deals) == 0:
            self.logger.warning(f"No trade deals found for any symbol in the last 72 hours.")
        else:
            deals_df = self._deals_to_dataframe(deals)
            # Filter for the specific symbol, entry deals, and bot's magic number
            entry_deals_df = deals_df[
                (deals_df['symbol'] == symbol) &
                (deals_df['entry'] == mt5.DEAL_ENTRY_IN) &
                (deals_df['magic'] == 20250824)
            ].copy()

            if entry_deals_df.empty:
                self.logger.warning(f"No entry trades found for {symbol} with magic number 20250824 in the last 72 hours.")
            else:
                self.logger.info(f"Found {len(entry_deals_df)} entry trades for {symbol}. Fetching order details...")
                # Fetch all historical orders in the date range to avoid calling per deal
                orders = mt5.history_orders_get(from_date, to_date)
                orders_df = self._orders_to_dataframe(orders) if orders else pd.DataFrame()

                for _, deal in entry_deals_df.iterrows():
                    trade_info = {
                        'entry_time': deal['time'],
                        'entry_price': deal['price'],
                        'trade_type': 'buy' if deal['type'] == mt5.ORDER_TYPE_BUY else 'sell',
                        'sl': 0,
                        'tp': 0
                    }
                    
                    # Find the corresponding order to get SL/TP
                    if not orders_df.empty:
                        order_ticket = deal['order']
                        order = orders_df[orders_df['ticket'] == order_ticket]
                        if not order.empty:
                            trade_info['sl'] = order.iloc[0]['sl']
                            trade_info['tp'] = order.iloc[0]['tp']
                        else:
                            self.logger.warning(f"Could not find order details for ticket {order_ticket}. Plotting without SL/TP.")
                    
                    trades_to_plot.append(trade_info)

        # --- Step 3: Prepare data and plot ---
        df_plot = df_entry.copy()
        df_plot['ema_entry'] = ta.ema(df_plot['close'], length=EMA_ENTRY_PERIOD)
        df_plot['ema_trend'] = ta.ema(df_plot['close'], length=EMA_TREND_PERIOD)

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(22, 10))
        ax.plot(df_plot.index, df_plot['close'], label="Price (M1)", color="blue", alpha=0.8, linewidth=1)
        ax.plot(df_plot.index, df_plot['ema_entry'], label=f'EMA {EMA_ENTRY_PERIOD} (M1)', color='orange', linestyle='--',
                linewidth=1.5)
        ax.plot(df_plot.index, df_plot['ema_trend'], label=f'EMA {EMA_TREND_PERIOD} (M1)', color='gray', linestyle='--',
                linewidth=1.5)

        # Plot trade entries
        for trade in trades_to_plot:
            entry_time = trade['entry_time']
            # Skip if trade is outside the chart's time range
            if entry_time < df_plot.index.min() or entry_time > df_plot.index.max():
                continue

            entry_price = trade['entry_price']
            trade_type = trade['trade_type']
            sl = trade['sl']
            tp = trade['tp']
            
            color = '#26a69a' if trade_type == 'buy' else '#ef5350'
            marker = '^' if trade_type == 'buy' else 'v'

            ax.scatter(
                entry_time, entry_price, color=color, s=self.trade_marker_size, marker=marker,
                edgecolors='black', zorder=5, label=f"{trade_type.upper()} Entry"
            )

            # Plot SL and TP lines
            if pd.notna(sl) and sl != 0 and pd.notna(tp) and tp != 0:
                ax.plot([entry_time, entry_time], [entry_price, tp], linestyle="--", color='green', alpha=0.9, linewidth=1.0, label="Take Profit")
                ax.plot([entry_time, entry_time], [entry_price, sl], linestyle="--", color='red', alpha=0.9, linewidth=1.0, label="Stop Loss")

        # --- Final Plot Configuration ---
        ax.set_title(f"Trade Analysis for {symbol} (Last 72 Hours from MT5)", fontsize=18)
        ax.set_xlabel("Time (UTC)", fontsize=12)
        ax.set_ylabel("Price", fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5)
        fig.tight_layout()

        # --- Save plot as PNG ---
        plot_filename = f"{symbol.replace(' ', '_').replace('(', '').replace(')', '')}_trade_analysis.png"
        full_plot_path = os.path.join(OUTPUT_DIR, plot_filename)
        fig.savefig(full_plot_path)
        plt.close(fig)
        self.logger.info(f"Chart has been saved as {plot_filename}. You can open this file in VS Code to view it.")


# --- Entry Point ---
# This block runs when the script is executed directly
if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="MT5 Trading Bot")
    parser.add_argument(
        '--mode', type=str, default='run', choices=['run', 'plot'],
        help="'run' to start trading, 'plot' to generate a chart."
    )
    parser.add_argument(
        '--symbol', type=str, default='Volatility 75 Index',
        help="The symbol to plot when in 'plot' mode."
    )
    parser.add_argument(
        '--max_trades', type=int, default=2,
        help="Maximum number of open trades per asset. Defaults to 2."
    )
    parser.add_argument(
        '--marker_size', type=int, default=25,
        help="Size of the buy/sell markers on the plot. Defaults to 25."
    )
    args = parser.parse_args()  # Parse the arguments provided at runtime

    # Create an instance of the trading bot
    bot = TradingBot(MT5_USER, MT5_PASS, MT5_SERVER, MT5_PATH, VIX_INDICES, args.max_trades, args.marker_size)

    if args.mode == 'run':
        # If mode is 'run', start the trading bot scheduler
        scheduler = BlockingScheduler(timezone=pytz.utc)  # Create a scheduler
        # Schedule the main bot loop to run every 1 minute
        scheduler.add_job(
            bot.run_bot_iteration, 'interval', minutes=1,
            start_date=datetime.now(pytz.utc) + timedelta(seconds=5)
        )
        logger.info("Starting MT5 Deriv Bot with Multi-Timeframe Strategy...")
        try:
            scheduler.start()  # Start the scheduler
        except (KeyboardInterrupt, SystemExit):
            # Handle graceful shutdown on user interruption (Ctrl+C)
            logger.info("Scheduler stopped by user. Disconnecting from MT5.")

    elif args.mode == 'plot':
        # If mode is 'plot', generate a chart for the specified symbol
        logger.info(f"--- Running in Plotting Mode for symbol: {args.symbol} ---")
        if bot.connect_mt5():  # Connect to MT5
            bot.plot_symbol_data(args.symbol)  # Generate the plot
            bot.disconnect_mt5()  # Disconnect when done
        else:
            logger.error("Cannot generate plot: MT5 connection failed.")