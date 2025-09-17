# -*- coding: utf-8 -*-
"""
Broker Client for MetaTrader 5.
"""

import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime
import time

import config
from notifier import Notifier, escape_markdown_v2

class BrokerClient:
    def __init__(self):
        """Initializes the MT5 connection and Notifier."""
        self.notifier = Notifier()
        if not mt5.initialize(login=config.MT5_LOGIN, password=config.MT5_PASSWORD, server=config.MT5_SERVER, path=config.MT5_PATH):
            error_message = f"MT5 initialization failed: {mt5.last_error()}"
            print(error_message)
            self.notifier.send_message(f"🚨 *Bot Error* 🚨\n{escape_markdown_v2(error_message)}")
            raise Exception("Broker connection failed.")
        
        account_info = mt5.account_info()
        if account_info is None:
            error_message = "Failed to get account info after MT5 initialization."
            print(error_message)
            self.notifier.send_message(f"🚨 *Bot Error* 🚨\n{escape_markdown_v2(error_message)}")
            raise Exception("Failed to get account info.")
            
        self.account_login = account_info.login
        print(f"Successfully connected to account #{self.account_login} on {account_info.server}")

    def shutdown(self):
        """Shuts down the connection to the MetaTrader 5 terminal."""
        print("Shutting down MT5 connection.")
        mt5.shutdown()

    def get_data(self, symbol, timeframe, start_date, end_date):
        """Fetches historical data from MT5 and converts it to the correct market timezone."""
        
        # Check if symbol is available
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Symbol {symbol} not found. Error: {mt5.last_error()}")
            return pd.DataFrame()

        if not symbol_info.visible:
            print(f"Symbol {symbol} is not visible in Market Watch, trying to enable it.")
            if not mt5.symbol_select(symbol, True):
                print(f"Failed to enable symbol {symbol}. Error: {mt5.last_error()}")
                return pd.DataFrame()
            # Wait a moment for the terminal to update
            time.sleep(1)

        utc_from = start_date
        utc_to = end_date
        
        rates = None
        for i in range(3): # Retry up to 3 times
            rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
            if rates is not None and len(rates) > 0:
                break
            print(f"Attempt {i+1} to fetch data for {symbol} failed. Retrying in 1 second...")
            time.sleep(1)

        if rates is None or len(rates) == 0:
            print(f"No data fetched for {symbol} from {start_date} to {end_date} after multiple retries. Last error: {mt5.last_error()}")
            return pd.DataFrame()
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')
        
        # Convert from UTC (MT5 server time) to the market's local timezone
        df.index = df.index.tz_localize('utc').tz_convert(config.MARKET_TIMEZONE)
        return df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'})

    def get_data_from_pos(self, symbol, timeframe, count):
        """Fetches OHLC data from the current position and returns a DataFrame."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            print(f"Error getting rates for {symbol} {timeframe}: {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df.index = df.index.tz_convert(config.MARKET_TIMEZONE)
        return df[['open', 'high', 'low', 'close']].rename(
            columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}
        )

    def execute_trade(self, action, symbol, lot_size, price, stop_loss, take_profit, magic):
        """Constructs and sends a trade request to MT5."""
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            error_message = f"Symbol {symbol} not found."
            print(error_message)
            self.notifier.send_message(f"⚠️ *Trade Failed* ⚠️\nSymbol: {escape_markdown_v2(symbol)}\nType: {escape_markdown_v2(action)}\nReason: {escape_markdown_v2(error_message)}")
            return None

        if not symbol_info.visible:
            print(f"Symbol {symbol} is not visible in Market Watch, trying to enable it.")
            if not mt5.symbol_select(symbol, True):
                print(f"Failed to enable symbol {symbol}. Error: {mt5.last_error()}")
            else:
                time.sleep(1) # Wait a moment for the terminal to update
                symbol_info = mt5.symbol_info(symbol) # Refresh symbol info

        if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            error_message = f"Trading for {symbol} is disabled. Please enable it in your terminal."
            print(error_message)
            self.notifier.send_message(f"⚠️ *Trade Failed* ⚠️\nSymbol: {escape_markdown_v2(symbol)}\nType: {escape_markdown_v2(action)}\nReason: {escape_markdown_v2(error_message)}")
            return None
        
        trade_type = mt5.TRADE_ACTION_DEAL
        point = symbol_info.point
        stops_level = symbol_info.trade_stops_level
        
        if action == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
            if price - stop_loss < stops_level * point:
                stop_loss = price - stops_level * point
                print(f"Adjusting SL for {symbol} BUY to {stop_loss}")
        elif action == "SELL":
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
            if stop_loss - price < stops_level * point:
                stop_loss = price + stops_level * point
                print(f"Adjusting SL for {symbol} SELL to {stop_loss}")
        else:
            return None

        # Normalize price and SL/TP to the correct number of digits
        digits = symbol_info.digits
        stop_loss = round(stop_loss, digits)
        take_profit = round(take_profit, digits)
        price = round(price, digits)

        request = {
            "action": trade_type,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "magic": magic,
            "comment": f"ORB_{action}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_message = f"Order send failed, retcode={result.retcode}"
            print(error_message)
            self.notifier.send_message(f"⚠️ *Trade Failed* ⚠️\nSymbol: {escape_markdown_v2(symbol)}\nType: {escape_markdown_v2(action)}\nReason: {escape_markdown_v2(result.comment)}")
            return None
        
        success_message = f"Successfully placed {action} order for {symbol} at {price} with SL={stop_loss}, TP={take_profit}"
        print(success_message)
        self.notifier.send_message(f"✅ *Trade Opened* ✅\nSymbol: {escape_markdown_v2(symbol)}\nType: {escape_markdown_v2(action)}\nEntry: {price:.5f}\nSL: {stop_loss:.5f}\nTP: {take_profit:.5f}\nVolume: {lot_size}")
        return result
