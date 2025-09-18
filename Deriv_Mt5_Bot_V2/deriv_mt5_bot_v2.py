# Import necessary libraries
import argparse
import csv
import logging
import os
import time
import requests
from datetime import datetime, timedelta
import numpy as np
import re
import xlsxwriter

import matplotlib.pyplot as plt
import pandas as pd
import pandas_ta as ta
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler

import MetaTrader5 as mt5
import deriv_mt5_bot_config as config
from deriv_mt5_bot_config import get_optimized_risk_parameters, get_telegram_credentials

# --- OUTPUT DIRECTORY ---
OUTPUT_DIR = "D:\\Microsoft VS Code\\Projects\\2025\\Deriv_EMA_Outputs"

# --- CONFIGURATION ---
credentials = config.get_credentials()
MT5_USER = credentials['user']
MT5_PASS = credentials['password']
MT5_SERVER = credentials['server']
MT5_PATH = credentials['path']

telegram_credentials = get_telegram_credentials()
TELEGRAM_BOT_TOKEN = telegram_credentials['bot_token']
TELEGRAM_CHAT_ID = telegram_credentials['chat_id']

VIX_INDICES = config.get_symbols()

# --- STRATEGY PARAMETERS ---
TF_TREND = config.TF_TREND
TF_ENTRY = config.TF_ENTRY
EMA_TREND_PERIOD = config.EMA_TREND_PERIOD
EMA_ENTRY_PERIOD = config.EMA_ENTRY_PERIOD
MAGIC_NUMBER = config.MAGIC_NUMBER

# --- INDICATOR SETTINGS ---
MACD_FAST = config.MACD_FAST
MACD_SLOW = config.MACD_SLOW
MACD_SIGNAL = config.MACD_SIGNAL
ATR_PERIOD = config.ATR_PERIOD
ADX_PERIOD = config.ADX_PERIOD

# --- RISK MANAGEMENT & STRATEGY ENHANCEMENTS ---
RISK_PERCENT_PER_TRADE = config.RISK_PERCENT_PER_TRADE
SL_MULTIPLIER = config.SL_MULTIPLIER
TP_RISK_REWARD = config.TP_RISK_REWARD
MAX_OPEN_TRADES_GLOBAL = config.MAX_OPEN_TRADES_GLOBAL
ADX_THRESHOLD = config.ADX_THRESHOLD

TRAILING_STOP_ATR_MULTIPLIER = config.TRAILING_STOP_ATR_MULTIPLIER
COMMISSION_PER_LOT = config.COMMISSION_PER_LOT

# --- CUSTOM SYMBOL SETTINGS ---
CUSTOM_LOT_SIZES = config.CUSTOM_LOT_SIZES
CUSTOM_MAX_OPEN_TRADES = config.CUSTOM_MAX_OPEN_TRADES
CUSTOM_RISK_PARAMETERS = config.CUSTOM_RISK_PARAMETERS

# --- Bot Configuration and Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "MT5_deriv_mt5_bot_v2_output.txt"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, user, password, server, path, trade_marker_size=25):
        self.logger = logging.getLogger(__name__)
        self.user = user
        self.password = password
        self.server = server
        self.path = path
        self.mt5_connected = False
        self.telegram_bot_token = TELEGRAM_BOT_TOKEN
        self.telegram_chat_id = TELEGRAM_CHAT_ID
        self.trade_marker_size = trade_marker_size # Add this line
        self.connect()

    def send_telegram_message(self, message):
        """Sends a message to the configured Telegram chat."""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            self.logger.warning("Telegram credentials are not set. Cannot send message.")
            return
        
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        payload = {
            'chat_id': self.telegram_chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                self.logger.info(f"Successfully sent Telegram message.")
            else:
                self.logger.error(f"Failed to send Telegram message. Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            self.logger.error(f"An exception occurred while sending Telegram message: {e}")

    def connect(self, max_retries=5, retry_delay=5):
        """Establishes a robust connection to MetaTrader 5 with retry logic."""
        for attempt in range(1, max_retries + 1):
            try:
                if mt5.initialize(path=self.path, login=self.user, password=self.password, server=self.server):
                    self.logger.info("MetaTrader 5 initialized successfully.")
                    account_info = mt5.account_info()
                    if account_info:
                        self.logger.info(f"Logged in to account #{account_info.login}")
                        self.mt5_connected = True
                        self.send_telegram_message(f"✅ *Bot Connected*\nAccount: {account_info.login}")
                        return True
                    else:
                        self.logger.error("Failed to get account info. Please check credentials.")
                        self.send_telegram_message("❌ *Bot Connection Failed*\nCould not get account info.")
                        self.shutdown()
                        return False
                else:
                    self.logger.warning(f"Failed to initialize MT5 on attempt {attempt}/{max_retries}. Last error: {mt5.last_error()}")
            except Exception as e:
                self.logger.error(f"An exception occurred during MT5 connection on attempt {attempt}/{max_retries}: {e}")

            if attempt < max_retries:
                self.logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        self.logger.critical("Failed to connect to MetaTrader 5 after several retries. Exiting.")
        self.send_telegram_message("❌ *Bot Connection Failed*\nCould not connect to MT5 after several retries.")
        self.mt5_connected = False
        return False

    def shutdown(self):
        """Shuts down the MetaTrader 5 connection."""
        self.logger.info("Shutting down MetaTrader 5 connection.")
        self.send_telegram_message("🛑 *Bot Shutting Down*")
        mt5.shutdown()
        self.mt5_connected = False

    def get_symbol_info(self, symbol):
        """Safely retrieves symbol information."""
        if not self.mt5_connected:
            self.logger.warning("Cannot get symbol info, not connected to MT5.")
            return None
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                self.logger.warning(f"Symbol {symbol} not found or not available.")
                return None
            return info
        except Exception as e:
            self.logger.error(f"Error getting info for symbol {symbol}: {e}")
            return None

    def get_candle_data(self, symbol, timeframe, count=100, max_retries=3, retry_delay=5):
        """
        Safely retrieves candle data from MT5 with retries and enhanced validation.
        """
        if not self.mt5_connected:
            self.logger.warning(f"Cannot get candle data for {symbol}, not connected to MT5.")
            return None

        # Ensure symbol is available
        if not mt5.symbol_select(symbol, True):
            self.logger.warning(f"Could not select {symbol} in Market Watch, it might be unavailable. Trying to fetch data anyway.")

        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            self.logger.error(f"Failed to get symbol info for {symbol}. Cannot download candle data.")
            return None

        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"Attempt {attempt}/{max_retries}: Starting download of {count} candles for {symbol} on timeframe {timeframe}...")
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

                if rates is None or len(rates) == 0:
                    self.logger.warning(f"No candle data returned for {symbol} on attempt {attempt}.")
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                        continue
                    else:
                        self.logger.error(f"Failed to download candle data for {symbol} after {max_retries} attempts.")
                        return None

                self.logger.info(f"Successfully downloaded {len(rates)} of {count} requested candles for {symbol}.")
                if len(rates) < count:
                    self.logger.warning(f"Downloaded fewer candles than requested for {symbol}. This may be due to limited history.")

                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
                df.sort_values(by='time', inplace=True)
                df.set_index('time', inplace=True)
                return df

            except Exception as e:
                self.logger.error(f"An exception occurred on attempt {attempt} while getting candle data for {symbol}: {e}", exc_info=True)
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    self.logger.critical(f"Failed to get candle data for {symbol} after all retries.")
                    return None
        return None

    def _get_live_signal(self, symbol):
        """Gets the latest data and calculates the trading signal."""
        df_m1 = self.get_candle_data(symbol, TF_ENTRY, count=EMA_TREND_PERIOD + 5)
        df_m5 = self.get_candle_data(symbol, TF_TREND, count=EMA_TREND_PERIOD + 5)

        if df_m1 is None or df_m5 is None or df_m1.empty or df_m5.empty:
            self.logger.warning(f"Could not get sufficient data for live signal on {symbol}.")
            return 'NONE', None

        df_m5['ema_trend'] = ta.ema(df_m5['close'], length=EMA_TREND_PERIOD)
        macd = ta.macd(df_m5['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        df_m5[f'macd'] = macd[f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
        df_m5[f'macds'] = macd[f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']
        
        last_m5 = df_m5.iloc[-1]
        trend = 'UP' if last_m5['close'] > last_m5['ema_trend'] and last_m5['macd'] > last_m5['macds'] else \
                'DOWN' if last_m5['close'] < last_m5['ema_trend'] and last_m5['macd'] < last_m5['macds'] else 'NONE'

        df_m1['ema_entry'] = ta.ema(df_m1['close'], length=EMA_ENTRY_PERIOD)
        df_m1['atr'] = ta.atr(df_m1['high'], df_m1['low'], df_m1['close'], length=ATR_PERIOD)
        adx = ta.adx(df_m1['high'], df_m1['low'], df_m1['close'], length=ADX_PERIOD)
        if adx is None or adx.empty or f'ADX_{ADX_PERIOD}' not in adx.columns:
            self.logger.warning(f"ADX calculation failed for {symbol}. Not enough data points.")
            return 'NONE', None
        df_m1['adx'] = adx[f'ADX_{ADX_PERIOD}']

        last = df_m1.iloc[-2]
        prev = df_m1.iloc[-3]

        atr_val = last['atr']
        if pd.isna(atr_val) or atr_val == 0:
            return 'NONE', None

        is_strong_trend = last['adx'] > ADX_THRESHOLD
        buy_crossover = prev['close'] <= prev['ema_entry'] and last['close'] > last['ema_entry']
        sell_crossover = prev['close'] >= prev['ema_entry'] and last['close'] < last['ema_entry']

        if trend == 'UP' and buy_crossover and is_strong_trend:
            return 'BUY', atr_val
        if trend == 'DOWN' and sell_crossover and is_strong_trend:
            return 'SELL', atr_val
            
        return 'NONE', None

    def _place_trade(self, symbol, trade_type, lot_size, sl_price, tp_price):
        """Places a trade on MetaTrader 5."""
        trade_type_str = "BUY" if trade_type == mt5.ORDER_TYPE_BUY else "SELL"
        self.logger.info(f"Attempting to place {trade_type_str} trade for {symbol} with lot size {lot_size}.")
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            self.logger.error(f"Failed to place trade for {symbol}, could not get symbol info.")
            return

        price = mt5.symbol_info_tick(symbol).ask if trade_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": trade_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "magic": MAGIC_NUMBER,
            "comment": "GeminiBot M5/M1",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order Send Failed for {symbol}. Retcode: {result.retcode}, Comment: {result.comment}")
            self.send_telegram_message(f"❌ *Trade Failed*\nSymbol: {symbol}\nType: {trade_type_str}\nError: {result.comment}")
        else:
            self.logger.info(f"Order Placed Successfully for {symbol}. Ticket: {result.order}")
            self.send_telegram_message(f"✅ *Trade Placed*\nSymbol: {symbol}\nType: {trade_type_str}\nPrice: {price:.5f}\nLots: {lot_size}\nSL: {sl_price:.5f}\nTP: {tp_price:.5f}")

    def _modify_trade(self, position, sl_price, tp_price):
        """Modifies the SL/TP of an open trade."""
        self.logger.info(f"Attempting to modify SL for position #{position.ticket}.")
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "sl": sl_price,
            "tp": tp_price,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Modify Failed for position #{position.ticket}. Retcode: {result.retcode}, Comment: {result.comment}")
            self.send_telegram_message(f"❌ *Modify Failed*\nPosition: {position.ticket}\nError: {result.comment}")
        else:
            self.logger.info(f"Modify Successful for position #{position.ticket}. New SL: {sl_price}")
            self.send_telegram_message(f"✅ *Trade Modified*\nPosition: {position.ticket}\nNew SL: {sl_price:.5f}")

    def check_for_trades(self):
        """The main logic loop for live trading."""
        self.logger.info("Checking for trading opportunities...")
        for symbol in VIX_INDICES:
            positions = mt5.positions_get(symbol=symbol)
            bot_positions = [p for p in positions if p.magic == MAGIC_NUMBER]

            max_trades_for_symbol = CUSTOM_MAX_OPEN_TRADES.get(symbol, MAX_OPEN_TRADES_GLOBAL)

            if len(bot_positions) < max_trades_for_symbol:
                signal, atr_val = self._get_live_signal(symbol)
                if signal != 'NONE':
                    direction_restriction = config.TRADE_DIRECTION_RESTRICTION.get(symbol)
                    if (signal == 'BUY' and direction_restriction == 'sells') or \
                       (signal == 'SELL' and direction_restriction == 'buys'):
                        self.logger.info(f"{signal} signal for {symbol} ignored due to trade direction restriction.")
                        continue

                    self.logger.info(f"New '{signal}' signal found for {symbol}.")
                    self.send_telegram_message(f"🔔 *New Signal*\nSymbol: {symbol}\nType: {signal}")
                    risk_params = CUSTOM_RISK_PARAMETERS.get(symbol, {})
                    sl_mult = risk_params.get('sl_multiplier', SL_MULTIPLIER)
                    tp_rr = risk_params.get('tp_risk_reward', TP_RISK_REWARD)

                    if risk_params:
                        self.logger.debug(f"Using custom risk parameters for {symbol}: SL Mult={sl_mult}, TP RR={tp_rr}")

                    sl_points = sl_mult * atr_val
                    
                    account_info = mt5.account_info()
                    if not account_info:
                        self.logger.error("Could not get account info to calculate lot size.")
                        continue
                    
                    symbol_info = self.get_symbol_info(symbol)
                    lot_size = self._calculate_lot_size(symbol, account_info.equity, sl_points, symbol_info)
                    if lot_size <= 0:
                        self.logger.warning(f"Calculated lot size is zero for {symbol}. Skipping trade.")
                        continue

                    tick = mt5.symbol_info_tick(symbol)
                    if signal == 'BUY':
                        trade_type = mt5.ORDER_TYPE_BUY
                        price = tick.ask
                        sl = price - sl_points
                        tp = price + (sl_points * tp_rr)
                    else: # SELL
                        trade_type = mt5.ORDER_TYPE_SELL
                        price = tick.bid
                        sl = price + sl_points
                        tp = price - (sl_points * tp_rr)
                    
                    self._place_trade(symbol, trade_type, lot_size, sl, tp)

            elif config.LIVE_TRADING_TRAILING_STOP and bot_positions:
                position = bot_positions[0]
                candles = self.get_candle_data(symbol, TF_ENTRY, count=20)
                if candles is None: continue
                
                atr = ta.atr(candles['high'], candles['low'], candles['close'], length=ATR_PERIOD).iloc[-1]
                
                if position.type == mt5.ORDER_TYPE_BUY:
                    new_sl = position.price_current - (atr * TRAILING_STOP_ATR_MULTIPLIER)
                    if new_sl > position.sl:
                        self._modify_trade(position, new_sl, position.tp)
                elif position.type == mt5.ORDER_TYPE_SELL:
                    new_sl = position.price_current + (atr * TRAILING_STOP_ATR_MULTIPLIER)
                    if new_sl < position.sl:
                        self._modify_trade(position, new_sl, position.tp)

    def run_live(self):
        """Runs the bot in live trading mode using a scheduler."""
        self.logger.info("Starting live trading scheduler.")
        self.send_telegram_message("▶️ *Live Trading Started*")
        scheduler = BlockingScheduler()
        scheduler.add_job(self.check_for_trades, 'cron', minute='*', second='5')
        
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self.logger.info("Live trading stopped by user.")
            self.send_telegram_message("⏹️ *Live Trading Stopped* by user.")

    def _prepare_data_for_backtest(self, symbol, days):
        self.logger.info(f"Preparing data for {symbol} for the last {days} days...")
        num_candles_m1 = days * 24 * 60
        df_m1 = self.get_candle_data(symbol, TF_ENTRY, count=num_candles_m1)
        if df_m1 is None or df_m1.empty or len(df_m1) < EMA_TREND_PERIOD:
            self.logger.warning(f"Not enough M1 data for {symbol}. Requested {num_candles_m1}, got {len(df_m1) if df_m1 is not None else 0}. Required at least {EMA_TREND_PERIOD}.")
            return pd.DataFrame()

        num_candles_m5 = days * 24 * 12
        df_m5 = self.get_candle_data(symbol, TF_TREND, count=num_candles_m5)
        if df_m5 is None or df_m5.empty or len(df_m5) < EMA_TREND_PERIOD:
            self.logger.warning(f"Not enough M5 data for {symbol}. Requested {num_candles_m5}, got {len(df_m5) if df_m5 is not None else 0}. Required at least {EMA_TREND_PERIOD}.")
            return pd.DataFrame()

        df_m5['ema_trend'] = ta.ema(df_m5['close'], length=EMA_TREND_PERIOD)
        macd = ta.macd(df_m5['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        df_m5[[f'macd', f'macds']] = macd[[f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}', f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}']]
        df_m5['trend'] = np.where((df_m5['close'] > df_m5['ema_trend']) & (df_m5['macd'] > df_m5['macds']), 'UP', 
                                np.where((df_m5['close'] < df_m5['ema_trend']) & (df_m5['macd'] < df_m5['macds']), 'DOWN', 'NONE'))

        df_m1['ema_entry'] = ta.ema(df_m1['close'], length=EMA_ENTRY_PERIOD)
        df_m1['atr'] = ta.atr(df_m1['high'], df_m1['low'], df_m1['close'], length=ATR_PERIOD)
        adx = ta.adx(df_m1['high'], df_m1['low'], df_m1['close'], length=ADX_PERIOD)
        if adx is None or adx.empty or f'ADX_{ADX_PERIOD}' not in adx.columns:
            self.logger.warning(f"Could not calculate ADX for {symbol} during data preparation.")
            return pd.DataFrame()
        df_m1['adx'] = adx[f'ADX_{ADX_PERIOD}']

        df_m1 = pd.merge_asof(df_m1.sort_index(), df_m5[['trend']].sort_index(), 
                              left_index=True, right_index=True, direction='backward')
        df_m1.dropna(inplace=True)
        self.logger.info(f"Data preparation complete for {symbol}. Shape: {df_m1.shape}")
        return df_m1

    def _calculate_lot_size(self, symbol, equity, sl_points, symbol_info):
        if not symbol_info:
            self.logger.warning(f"Cannot calculate lot size for {symbol}, no symbol info available.")
            return 0.0

        volume_min = symbol_info.volume_min
        volume_max = symbol_info.volume_max
        volume_step = symbol_info.volume_step

        lot_size = 0.0

        if symbol in CUSTOM_LOT_SIZES:
            lot_size = CUSTOM_LOT_SIZES[symbol]
            self.logger.debug(f"Using custom lot size {lot_size} for {symbol}.")
        else:
            lot_size = volume_min
            self.logger.debug(f"No custom lot size for {symbol}, using minimum lot size {lot_size}.")

        if lot_size < volume_min:
            self.logger.warning(f"Custom lot size {lot_size} for {symbol} is below the minimum of {volume_min}. Adjusting to minimum.")
            lot_size = volume_min
        
        if lot_size > volume_max:
            self.logger.warning(f"Custom lot size {lot_size} for {symbol} is above the maximum of {volume_max}. Adjusting to maximum.")
            lot_size = volume_max

        if volume_step > 0:
            lot_size = round(lot_size / volume_step) * volume_step
        
        return round(lot_size, 2)

    def _run_vectorized_backtest(self, data, symbol, sl_multiplier, tp_risk_reward, use_trailing_stop, initial_capital=100):
        simulated_trades = []
        open_trades = []
        equity = initial_capital
        
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            self.logger.error(f"Could not run backtest for {symbol}, failed to get symbol info.")
            return pd.DataFrame()

        max_trades_for_symbol = CUSTOM_MAX_OPEN_TRADES.get(symbol, MAX_OPEN_TRADES_GLOBAL)
        self.logger.debug(f"Backtest for {symbol} will use a max of {max_trades_for_symbol} open trade(s).")

        spread = (symbol_info.ask - symbol_info.bid)
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        value_per_point = tick_value / tick_size if tick_size > 0 else 0
        if value_per_point == 0:
            self.logger.error(f"Could not run backtest for {symbol}, value per point is zero.")
            return pd.DataFrame()

        strong_trend = data['adx'] > ADX_THRESHOLD
        buy_crossover = (data['close'].shift(1) <= data['ema_entry'].shift(1)) & (data['close'] > data['ema_entry'])
        sell_crossover = (data['close'].shift(1) >= data['ema_entry'].shift(1)) & (data['close'] < data['ema_entry'])
        buy_signals = (data['trend'] == 'UP') & buy_crossover & strong_trend
        sell_signals = (data['trend'] == 'DOWN') & sell_crossover & strong_trend

        for i in range(1, len(data)):
            candle = data.iloc[i]
            
            for trade in open_trades[:]: 
                exit_condition_met = False
                pnl_points = 0

                if trade['type'] == 'buy':
                    if use_trailing_stop:
                        new_sl = candle['close'] - (data['atr'].iloc[i] * TRAILING_STOP_ATR_MULTIPLIER)
                        if new_sl > trade['sl']:
                            trade['sl'] = new_sl
                    if candle['low'] <= trade['sl']:
                        pnl_points = trade['sl'] - trade['entry_price']
                        trade.update({'outcome': 'LOSS', 'exit_price': trade['sl'], 'exit_time': candle.name})
                        exit_condition_met = True
                    elif not use_trailing_stop and candle['high'] >= trade['tp']:
                        pnl_points = trade['tp'] - trade['entry_price']
                        trade.update({'outcome': 'WIN', 'exit_price': trade['tp'], 'exit_time': candle.name})
                        exit_condition_met = True
                elif trade['type'] == 'sell':
                    if use_trailing_stop:
                        new_sl = candle['close'] + (data['atr'].iloc[i] * TRAILING_STOP_ATR_MULTIPLIER)
                        if new_sl < trade['sl']:
                            trade['sl'] = new_sl
                    if candle['high'] >= trade['sl']:
                        pnl_points = trade['entry_price'] - trade['sl']
                        trade.update({'outcome': 'LOSS', 'exit_price': trade['sl'], 'exit_time': candle.name})
                        exit_condition_met = True
                    elif not use_trailing_stop and candle['low'] <= trade['tp']:
                        pnl_points = trade['entry_price'] - trade['tp']
                        trade.update({'outcome': 'WIN', 'exit_price': trade['tp'], 'exit_time': candle.name})
                        exit_condition_met = True
                
                if exit_condition_met:
                    pnl_dollars = (pnl_points * value_per_point * trade['lot_size']) - (COMMISSION_PER_LOT * trade['lot_size'])
                    trade['pnl'] = pnl_dollars
                    equity += pnl_dollars
                    simulated_trades.append(trade)
                    open_trades.remove(trade)

            if len(open_trades) < max_trades_for_symbol:
                signal = 'buy' if buy_signals.iloc[i] else 'sell' if sell_signals.iloc[i] else None
                if signal:
                    direction_restriction = config.TRADE_DIRECTION_RESTRICTION.get(symbol)
                    if (signal == 'buy' and direction_restriction == 'sells') or \
                       (signal == 'sell' and direction_restriction == 'buys'):
                        continue

                    if any(t['entry_time'] == data.index[i] for t in open_trades):
                        continue

                    atr_val = data['atr'].iloc[i]
                    if pd.isna(atr_val) or atr_val == 0: continue
                    
                    sl_points = sl_multiplier * atr_val
                    lot_size = self._calculate_lot_size(symbol, equity, sl_points, symbol_info)
                    if lot_size <= 0: continue
                    
                    if signal == 'buy':
                        entry_price = data['close'].iloc[i] + spread
                        sl = entry_price - sl_points
                        tp = entry_price + (tp_risk_reward * sl_points)
                    else:
                        entry_price = data['close'].iloc[i]
                        sl = entry_price + sl_points
                        tp = entry_price - (tp_risk_reward * sl_points)

                    new_trade = {
                        'symbol': symbol, 'type': signal, 'entry_price': entry_price,
                        'entry_time': data.index[i], 'sl': sl, 'tp': tp, 'lot_size': lot_size, 'outcome': 'OPEN'
                    }
                    open_trades.append(new_trade)
        
        simulated_trades.sort(key=lambda x: x['entry_time'])
        return pd.DataFrame(simulated_trades)

    def calculate_performance_metrics(self, trades_df, initial_capital=100):
        if trades_df.empty:
            return {'pnl': 0, 'equity_curve': [initial_capital]}

        trades_df['pnl'] = pd.to_numeric(trades_df['pnl'], errors='coerce')
        trades_df.dropna(subset=['pnl'], inplace=True)

        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        win_rate = (len(wins) / len(trades_df)) * 100 if not trades_df.empty else 0
        gross_profit = wins['pnl'].sum()
        gross_loss = abs(losses['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        net_pnl = trades_df['pnl'].sum()
        
        trades_df.sort_values(by='exit_time', inplace=True)
        equity_curve = (initial_capital + trades_df['pnl'].cumsum()).tolist()
        equity_curve.insert(0, initial_capital)
        
        equity_df = pd.DataFrame(equity_curve, columns=['equity'])
        peak = equity_df['equity'].expanding(min_periods=1).max()
        drawdown_dollars = peak - equity_df['equity']
        drawdown_pct = (drawdown_dollars / peak) * 100
        max_drawdown_dollars = drawdown_dollars.max() 
        max_drawdown_pct = drawdown_pct.max()
        
        daily_returns = trades_df.set_index('exit_time')['pnl'].resample('D').sum()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) if daily_returns.std() > 0 else 0

        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std(ddof=0) if not downside_returns.empty else 0
        sortino_ratio = (daily_returns.mean() / downside_deviation) * np.sqrt(365) if downside_deviation > 0 else 0

        total_return = net_pnl / initial_capital
        num_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
        if num_days > 0:
            annualized_return = ((1 + total_return) ** (365.0 / num_days)) - 1
            if abs(max_drawdown_pct) > 0:
                calmar_ratio = annualized_return / (abs(max_drawdown_pct) / 100)
            else:
                calmar_ratio = 0
        else:
            calmar_ratio = 0

        avg_holding_time = (trades_df['exit_time'] - trades_df['entry_time']).mean()

        is_win = trades_df['pnl'] > 0
        consecutive_streaks = is_win.groupby((is_win != is_win.shift()).cumsum()).cumcount() + 1
        max_consecutive_wins = consecutive_streaks[is_win].max() if not consecutive_streaks[is_win].empty else 0
        max_consecutive_losses = consecutive_streaks[~is_win].max() if not consecutive_streaks[~is_win].empty else 0

        biggest_profit = wins['pnl'].max() if not wins.empty else 0
        biggest_loss = losses['pnl'].min() if not losses.empty else 0

        return {
            'total_trades': len(trades_df),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown_pct,
            'max_drawdown_dollars': max_drawdown_dollars,
            'sharpe_ratio': sharpe_ratio,
            'pnl': net_pnl,
            'equity_curve': equity_curve,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'avg_holding_time': str(avg_holding_time),
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'biggest_profit': biggest_profit,
            'biggest_loss': biggest_loss
        }

    def run_backtest(self, symbol, days, sl_multiplier, tp_risk_reward, initial_capital=100, generate_report=True, generate_plot=False):
        self.logger.info(f"--- Starting single backtest for {symbol} ---")
        data = self._prepare_data_for_backtest(symbol, days)
        if data.empty:
            self.logger.error(f"Cannot run backtest for {symbol}, data preparation failed.")
            return None

        trades_df = self._run_vectorized_backtest(
            data,
            symbol,
            sl_multiplier,
            tp_risk_reward,
            use_trailing_stop=config.BACKTEST_TRAILING_STOP,
            initial_capital=initial_capital
        )
        
        if trades_df.empty:
            self.logger.warning(f"No trades were executed for {symbol} in the backtest.")
            performance = self.calculate_performance_metrics(trades_df, initial_capital)
        else:
            performance = self.calculate_performance_metrics(trades_df, initial_capital)

        self.logger.info(f"--- Backtest Results for {symbol} ---")
        self.logger.info(f"Total Trades: {performance.get('total_trades', 0)}")
        self.logger.info(f"Win Rate: {performance.get('win_rate', 0):.2f}%")
        self.logger.info(f"Profit Factor: {performance.get('profit_factor', 0):.2f}")
        self.logger.info(f"Net PnL: ${performance.get('pnl', 0):.2f}")
        self.logger.info(f"Max Drawdown: ${performance.get('max_drawdown_dollars', 0):.2f} ({performance.get('max_drawdown_pct', 0):.2f}%)")
        self.logger.info(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
        
        if generate_report:
            backtest_params = {
                'symbol': symbol,
                'sl_multiplier': sl_multiplier,
                'tp_risk_reward': tp_rr
            }
            self._generate_excel_report(performance, trades_df, backtest_params, initial_capital=initial_capital, days_tested=days, trailing_stop_status=config.BACKTEST_TRAILING_STOP)

        if generate_plot and not trades_df.empty:
            self._plot_backtest_results(data, trades_df, symbol)

        return performance

    def _plot_backtest_results(self, price_data, trades_df, symbol):
        self.logger.info(f"Generating backtest plot for {symbol}...")
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(15, 8))

        ax.plot(price_data.index, price_data['close'], label='Close Price', color='skyblue', linewidth=1)
        ax.plot(price_data.index, price_data['ema_entry'], label=f'EMA({EMA_ENTRY_PERIOD})', color='orange', linestyle='--',
                linewidth=1)

        for _, trade in trades_df.iterrows():
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            entry_price = trade['entry_price']
            sl = trade['sl']
            tp = trade['tp']
            
            color = 'green' if trade['type'] == 'buy' else 'red'
            marker = '^' if trade['type'] == 'buy' else 'v'
            
            ax.plot(entry_time, entry_price, marker=marker, color=color, markersize=8, label=f'{trade["type"].upper()} Entry')
            
            ax.plot([entry_time, entry_time], [entry_price, sl], color='red', linestyle=':', linewidth=1)
            ax.plot([entry_time, entry_time], [entry_price, tp], color='green', linestyle=':', linewidth=1)

        ax.set_title(f'Backtest Results for {symbol}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = os.path.join(OUTPUT_DIR, f"{symbol.replace('/', '_')}_backtest_chart.png")
        plt.savefig(filename)
        self.logger.info(f"Backtest plot saved to {filename}")
        plt.close(fig)

    def _generate_excel_report(self, performance, trades_df, params, filename="MT5_trading_results.xlsx", symbol_performance_data=None, initial_capital=100, days_tested=None, trailing_stop_status=None):
        full_filename = os.path.join(OUTPUT_DIR, filename)
        self.logger.info(f"Generating Excel report to {full_filename}...")
        try:
            with pd.ExcelWriter(full_filename, engine='xlsxwriter') as writer:
                performance_summary = {k: v for k, v in performance.items() if k != 'equity_curve'}
                performance_summary['Initial Capital'] = initial_capital
                
                # Add new report details
                report_generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                performance_summary['Days Tested'] = days_tested
                performance_summary['Trailing Stoploss'] = "On" if trailing_stop_status else "Off"
                performance_summary['Report Generated On'] = report_generated_time
                performance_summary['Biggest Profit'] = performance.get('biggest_profit', 0)
                performance_summary['Biggest Loss'] = performance.get('biggest_loss', 0)

                summary_df = pd.DataFrame([performance_summary])
                summary_df.to_excel(writer, sheet_name='Backtest_Report', index=False)
                
                if not trades_df.empty:
                    # Convert timezone-aware datetime columns to timezone-naive
                    if 'entry_time' in trades_df.columns and pd.api.types.is_datetime64_any_dtype(trades_df['entry_time']):
                        trades_df['entry_time'] = trades_df['entry_time'].dt.tz_localize(None)
                    if 'exit_time' in trades_df.columns and pd.api.types.is_datetime64_any_dtype(trades_df['exit_time']):
                        trades_df['exit_time'] = trades_df['exit_time'].dt.tz_localize(None)
                    trades_df.to_excel(writer, sheet_name='Trade_Log', index=False)

                self._write_backtest_summary_sheet(writer, performance, initial_capital)

                if symbol_performance_data:
                    for spd in symbol_performance_data:
                        spd['Initial Capital'] = initial_capital
                    symbol_perf_df = pd.DataFrame(symbol_performance_data)
                    symbol_perf_df.to_excel(writer, sheet_name='Symbol_Performance', index=False)

                workbook = writer.book
                worksheet = writer.sheets['Backtest_Report']
                
                equity_curve = performance.get('equity_curve', [])
                if equity_curve and not trades_df.empty:
                    equity_df = pd.DataFrame({
                        'Time': [trades_df['entry_time'].iloc[0]] + trades_df['exit_time'].tolist(),
                        'Equity': equity_curve
                    })
                    equity_df.to_excel(writer, sheet_name='EquityCurveData', index=False)
                    
                    chart = workbook.add_chart({'type': 'line'})
                    chart.add_series({
                        'name':       'Equity Curve',
                        'categories': f'=EquityCurveData!$A$2:$A${len(equity_curve)}',
                        'values':     f'=EquityCurveData!$B$2:$B${len(equity_curve)}',
                    })
                    chart.set_title({'name': 'Equity Curve'})
                    chart.set_x_axis({'name': 'Time', 'date_axis': True})
                    chart.set_y_axis({'name': 'Account Equity ($)'})
                    chart.set_legend({'position': 'none'})
                    worksheet.insert_chart('A5', chart, {'x_scale': 2, 'y_scale': 1.5})

            self.logger.info(f"Successfully saved Excel report: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to generate Excel report: {e}", exc_info=True)

    def _write_backtest_summary_sheet(self, writer, performance, initial_capital):
        summary_sheet = writer.book.add_worksheet('Summary')
        bold = writer.book.add_format({'bold': True})
        summary_sheet.set_column('A:B', 100)

        pnl = performance.get('pnl', 0)
        win_rate = performance.get('win_rate', 0)
        profit_factor = performance.get('profit_factor', 0)
        drawdown_pct = performance.get('max_drawdown_pct', 0)
        sharpe_ratio = performance.get('sharpe_ratio', 0)

        summary_sheet.write('A1', 'Data-Driven Backtest Analysis', bold)
        summary_sheet.write('A2', f"Initial Capital: ${initial_capital:.2f}")

        summary_sheet.write('A4', '1. Risk Assessment', bold)
        risk_level = "High"
        if drawdown_pct < 10:
            risk_level = "Low"
        elif drawdown_pct < 25:
            risk_level = "Moderate"
        summary_sheet.write('A4', f"The maximum drawdown was {drawdown_pct:.2f}%. This is considered a {risk_level} level of risk.")

        summary_sheet.write('A6', '2. Profitability Confirmation', bold)
        profitability = "Unprofitable"
        if profit_factor > 1.5:
            profitability = "Strong"
        elif profit_factor > 1:
            profitability = "Modest"
        summary_sheet.write('A7', f"The strategy was { 'profitable' if pnl > 0 else 'unprofitable'} with a final PnL of ${pnl:.2f}. The profit factor of {profit_factor:.2f} indicates {profitability} profitability.")

        summary_sheet.write('A9', '3. Consistency Check', bold)
        consistency = "Poor"
        if sharpe_ratio > 1.0:
            consistency = "Excellent"
        elif sharpe_ratio > 0.5:
            consistency = "Good"
        elif sharpe_ratio > 0:
            consistency = "Acceptable"
        summary_sheet.write('A10', f"Consistency appears {consistency}. The Sharpe Ratio of {sharpe_ratio:.2f} indicates the risk-adjusted return. The win rate was {win_rate:.2f}%. ")

        summary_sheet.write('A12', '4. Balanced Verdict', bold)
        verdict = ""
        if pnl > 0 and sharpe_ratio > 0.8 and drawdown_pct < 15:
            verdict = "Excellent. The strategy shows strong, consistent, risk-adjusted returns with low drawdown. It appears robust."
        elif pnl > 0 and sharpe_ratio > 0.5 and drawdown_pct < 25:
            verdict = "Good. The strategy is profitable and reasonably consistent. The risk level is moderate. Consider deploying, but monitor performance closely."
        elif pnl > 0:
            verdict = "Acceptable, with caveats. The strategy is profitable, but may have low consistency (Sharpe Ratio < 0.5) or high risk (Drawdown > 25%). Proceed with caution."
        else:
            verdict = "Not Recommended. The strategy was unprofitable in its current configuration for the tested period."
        summary_sheet.write('A13', verdict)

        summary_sheet.write('A15', '5. Advanced Metrics', bold)
        
        sortino = performance.get('sortino_ratio', 0)
        calmar = performance.get('calmar_ratio', 0)
        holding_time = performance.get('avg_holding_time', 'N/A')
        max_wins = performance.get('max_consecutive_wins', 0)
        max_losses = performance.get('max_consecutive_losses', 0)

        summary_sheet.write('A16', f"Sortino Ratio: {sortino:.2f} (Measures return against downside risk)")
        summary_sheet.write('A17', f"Calmar Ratio: {calmar:.2f} (Measures return against max drawdown)")
        summary_sheet.write('A18', f"Average Trade Holding Time: {holding_time}")
        summary_sheet.write('A19', f"Max Consecutive Wins: {max_wins}")
        summary_sheet.write('A20', f"Max Consecutive Losses: {max_losses}")

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

    def plot_live_trades(self, symbol, plot_hours=72):
        """
        Fetches required data from MT5, plots the chart with live trades, and saves it as a PNG file.
        """
        self.logger.info(f"Preparing detailed plot for {symbol} from MT5 data...")

        # --- Step 1: Fetch price data ---
        num_candles = plot_hours * 60  # hours * 60 minutes/hour
        self.logger.info(f"Fetching latest {num_candles} M1 candles for {symbol} (approx. {plot_hours} hours).")
        df_entry = self.get_candle_data(symbol, TF_ENTRY, count=num_candles)
        if df_entry is None or df_entry.empty:
            self.logger.error(f"No M1 data available for {symbol}. Cannot generate plot.")
            return

        # --- Step 2: Fetch Trade History from MT5 ---
        self.logger.info(f"Fetching trade history from MT5 for the last {plot_hours} hours.")
        from_date = datetime.now(pytz.utc) - timedelta(hours=plot_hours)
        to_date = datetime.now(pytz.utc)
        deals = mt5.history_deals_get(from_date, to_date)
        
        trades_to_plot = []
        if deals is None or len(deals) == 0:
            self.logger.warning(f"No trade deals found for any symbol in the last {plot_hours} hours.")
        else:
            deals_df = self._deals_to_dataframe(deals)
            # Filter for the specific symbol, entry deals, and bot's magic number
            entry_deals_df = deals_df[
                (deals_df['symbol'] == symbol) &
                (deals_df['entry'] == mt5.DEAL_ENTRY_IN) &
                (deals_df['magic'] == MAGIC_NUMBER)
            ].copy()

            if entry_deals_df.empty:
                self.logger.warning(f"No entry trades found for {symbol} with magic number {MAGIC_NUMBER} in the last {plot_hours} hours.")
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
        ax.set_title(f"Trade Analysis for {symbol} (Last {plot_hours} Hours from MT5)", fontsize=18)
        ax.set_xlabel("Time (UTC)", fontsize=12)
        ax.set_ylabel("Price", fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5)
        fig.tight_layout()

        # --- Save plot as PNG ---
        # Define the directory where plots will be saved. Assuming project root for now.
        # You might want to make this configurable or create a 'plots' subdirectory.
        plot_dir = OUTPUT_DIR
        os.makedirs(plot_dir, exist_ok=True) # Ensure the directory exists

        plot_filename_base = f"{symbol.replace(' ', '_').replace('(', '').replace(')', '')}_trade_analysis"
        
        # Generate a filename and save the plot, overwriting if it exists.
        plot_filename = f"{plot_filename_base}.png"
        full_plot_path = os.path.join(plot_dir, plot_filename)
        
        fig.savefig(full_plot_path)
        plt.close(fig)
        self.logger.info(f"Chart has been saved as {full_plot_path}. You can open this file in VS Code to view it.")


    def run_portfolio_backtest(self, days, initial_capital=100):
        self.logger.info(f"--- Starting portfolio backtest over {days} days ---")
        all_portfolio_trades = []
        all_symbol_performance = []
        
        symbols_to_backtest = config.get_symbols()
        if not symbols_to_backtest:
            self.logger.error("No symbols defined in config.get_symbols() for portfolio backtest. Aborting.")
            return None

        for symbol in symbols_to_backtest:
            self.logger.info(f"Running backtest for individual symbol: {symbol}")
            
            risk_params = CUSTOM_RISK_PARAMETERS.get(symbol, {})
            sl_mult = risk_params.get('sl_multiplier', SL_MULTIPLIER)
            tp_rr = risk_params.get('tp_risk_reward', TP_RISK_REWARD)

            self.logger.info(f"  Using SL Mult={sl_mult}, TP RR={tp_rr} for {symbol}")

            data = self._prepare_data_for_backtest(symbol, days)
            if data.empty:
                self.logger.warning(f"Skipping {symbol} due to data preparation failure.")
                continue

            symbol_trades_df = self._run_vectorized_backtest(
                data=data,
                symbol=symbol,
                sl_multiplier=sl_mult,
                tp_risk_reward=tp_rr,
                use_trailing_stop=config.BACKTEST_TRAILING_STOP,
                initial_capital=initial_capital
            )
            
            if symbol_trades_df is not None and not symbol_trades_df.empty:
                all_portfolio_trades.append(symbol_trades_df)
                symbol_performance = self.calculate_performance_metrics(symbol_trades_df, initial_capital)
                symbol_performance['symbol'] = symbol
                all_symbol_performance.append(symbol_performance)
            else:
                self.logger.warning(f"No trades generated for {symbol} in portfolio backtest.")

        if not all_portfolio_trades:
            self.logger.warning("No trades generated across all symbols for portfolio backtest.")
            return None

        portfolio_trades_df = pd.concat(all_portfolio_trades, ignore_index=True)
        portfolio_trades_df.sort_values(by='entry_time', inplace=True)
        
        self.logger.info(f"Total trades in portfolio backtest: {len(portfolio_trades_df)}")

        portfolio_performance = self.calculate_performance_metrics(portfolio_trades_df, initial_capital)
        
        self.logger.info(f"--- Portfolio Backtest Results ---")
        self.logger.info(f"Total Trades: {portfolio_performance.get('total_trades', 0)}")
        self.logger.info(f"Net PnL: ${portfolio_performance.get('pnl', 0):.2f}")
        self.logger.info(f"Max Drawdown: ${portfolio_performance.get('max_drawdown_dollars', 0):.2f} ({portfolio_performance.get('max_drawdown_pct', 0):.2f}%)")

        portfolio_report_filename = "MT5_portfolio_results.xlsx"
        self._generate_excel_report(
            performance=portfolio_performance, 
            trades_df=portfolio_trades_df, 
            params={},
            filename=portfolio_report_filename,
            symbol_performance_data=all_symbol_performance,
            initial_capital=initial_capital,
            days_tested=days,
            trailing_stop_status=config.BACKTEST_TRAILING_STOP
        )

        return portfolio_performance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deriv MT5 Trading Bot")
    parser.add_argument('--mode', type=str, default='live', choices=['live', 'backtest', 'portfolio_backtest', 'plot_live'],
                        help='The mode to run the bot in: live, backtest, portfolio_backtest, or plot_live.')
    parser.add_argument('--symbol', type=str, default='Volatility 75 Index', help='The symbol to backtest or plot.')
    parser.add_argument('--days', type=int, default=30, help='The number of days to backtest.')
    parser.add_argument('--portfolio_days', type=int, default=30, help='The number of days to backtest for the portfolio.')
    parser.add_argument('--plot', action='store_true', help='Generate a plot of the backtest results.')
    parser.add_argument(
        '--marker_size',
        type=int,
        default=25,
        help="Size of the buy/sell markers on the plot. Defaults to 25."
    )
    parser.add_argument(
        '--plot_hours',
        type=int,
        default=48,
        help="Number of hours to plot for live trade analysis. Defaults to 48."
    )
    args = parser.parse_args()

    bot = None
    try:
        bot = TradingBot(user=MT5_USER, password=MT5_PASS, server=MT5_SERVER, path=MT5_PATH, trade_marker_size=args.marker_size)
        
        if bot.mt5_connected:
            if args.mode == 'backtest':
                risk_params = CUSTOM_RISK_PARAMETERS.get(args.symbol, {})
                sl_mult = risk_params.get('sl_multiplier', SL_MULTIPLIER)
                tp_rr = risk_params.get('tp_risk_reward', TP_RISK_REWARD)

                logger.info(f"Running backtest for {args.symbol} with SL Mult={sl_mult} and TP RR={tp_rr}")

                bot.run_backtest(
                    symbol=args.symbol,
                    days=args.days,
                    sl_multiplier=sl_mult,
                    tp_risk_reward=tp_rr,
                    generate_plot=args.plot
                )
            elif args.mode == 'live':
                bot.run_live()
            elif args.mode == 'portfolio_backtest':
                bot.run_portfolio_backtest(days=args.portfolio_days)
            elif args.mode == 'plot_live':
                logger.info(f"--- Running in Live Plotting Mode for symbol: {args.symbol} for last {args.plot_hours} hours ---")
                bot.plot_live_trades(args.symbol, plot_hours=args.plot_hours)
        else:
            logger.critical("Bot could not connect to MT5. Aborting.")

    except KeyboardInterrupt:
        logger.info("Bot execution interrupted by user.")
    except Exception as e:
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)
        if bot:
            bot.send_telegram_message(f"🚨 *CRITICAL ERROR*\nBot is shutting down due to an unhandled exception: {e}")
    finally:
        if bot:
            bot.shutdown()

