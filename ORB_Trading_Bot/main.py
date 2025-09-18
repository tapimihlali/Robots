# -*- coding: utf-8 -*-
"""
Main entry point for the ORB Trading Bot.
"""

import argparse
import time
from datetime import datetime, time as time_obj, timedelta
import pytz
from notifier import escape_markdown_v2
import sys
import os
import MetaTrader5 as mt5

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure timely output
    def flush(self):
        for f in self.files:
            f.flush()

# --- Output Redirection Setup ---
output_dir = r'D:\Microsoft VS Code\Projects\2025\Robots\ORB_Outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

log_file_path = os.path.join(output_dir, 'ORB_Main_output.txt')
log_file = open(log_file_path, 'w')
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, log_file)
# --- End Output Redirection Setup ---

from api_client import BrokerClient
import strategy
import config

def get_symbol_market_type(symbol):
    if symbol in config.US_SYMBOLS:
        return 'US'
    if symbol in config.EUROPEAN_SYMBOLS:
        return 'EUROPEAN'
    if symbol in config.FOREX_SYMBOLS or symbol in config.CRYPTO_SYMBOLS or symbol in config.COMMODITY_SYMBOLS:
        return '24_HOUR'
    return None

def get_timeframe_in_seconds(timeframe):
    """Converts an MT5 timeframe constant to seconds."""
    if timeframe == mt5.TIMEFRAME_M1:
        return 60
    if timeframe == mt5.TIMEFRAME_M5:
        return 300
    if timeframe == mt5.TIMEFRAME_M15:
        return 900
    if timeframe == mt5.TIMEFRAME_M30:
        return 1800
    if timeframe == mt5.TIMEFRAME_H1:
        return 3600
    # Add other timeframes as needed
    return 60 # Default to 1 minute

def get_next_market_open(market_type, current_time):
    """Gets the next market open time for a given market type."""
    if market_type == 'US':
        open_time = time_obj.fromisoformat(config.US_MARKET_OPEN_TIME)
        next_open = current_time.replace(hour=open_time.hour, minute=open_time.minute, second=0, microsecond=0)
        if current_time.time() >= open_time:
            next_open += timedelta(days=1)
        return next_open
    elif market_type == 'EUROPEAN':
        open_time = time_obj.fromisoformat(config.EUROPEAN_MARKET_OPEN_TIME)
        next_open = current_time.replace(hour=open_time.hour, minute=open_time.minute, second=0, microsecond=0)
        if current_time.time() >= open_time:
            next_open += timedelta(days=1)
        return next_open
    elif market_type == '24_HOUR':
        next_opens = []
        for session, open_time_str in config.SESSION_OPEN_TIMES.items():
            open_time = time_obj.fromisoformat(open_time_str)
            next_open = current_time.replace(hour=open_time.hour, minute=open_time.minute, second=0, microsecond=0)
            if current_time.time() >= open_time:
                next_open += timedelta(days=1)
            next_opens.append(next_open)
        return min(next_opens)
    return None

def calculate_lot_size(symbol, stop_loss_pips):
    """Calculates the lot size based on the configured strategy."""
    # 1. Check for custom lot size
    if symbol in config.CUSTOM_LOT_SIZES:
        return config.CUSTOM_LOT_SIZES[symbol]

    # 2. Global lot size strategy
    if config.DEFAULT_LOT_SIZE_STRATEGY == "FIXED":
        return config.LOT_SIZE
    
    elif config.DEFAULT_LOT_SIZE_STRATEGY == "MIN_LOT":
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            return symbol_info.volume_min
        else:
            return config.LOT_SIZE # Fallback

    elif config.DEFAULT_LOT_SIZE_STRATEGY == "RISK_PERCENT":
        account_info = mt5.account_info()
        if account_info:
            equity = account_info.equity
            risk_amount = equity * (config.RISK_PER_TRADE_PERCENT / 100)
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                tick_value = symbol_info.trade_tick_value
                if tick_value > 0 and stop_loss_pips > 0:
                    lot_size = risk_amount / (stop_loss_pips * tick_value)
                    
                    # Normalize and validate the lot size
                    min_volume = symbol_info.volume_min
                    max_volume = symbol_info.volume_max
                    volume_step = symbol_info.volume_step
                    
                    if lot_size < min_volume:
                        lot_size = min_volume
                    if lot_size > max_volume:
                        lot_size = max_volume
                    
                    lot_size = round(lot_size / volume_step) * volume_step
                    return round(lot_size, 2)

    return config.LOT_SIZE # Default fallback

def run_live(client):
    """Runs the bot in a continuous loop for live trading."""
    print("\n--- Starting Live Trading Bot ---")
    
    states = {symbol: {'orh': None, 'orl': None, 'last_checked_day': None, 'trade_taken_today': False, 'bullish_break': False, 'bearish_break': False} for symbol in config.SYMBOLS}
    
    while True:
        try:
            market_tz = pytz.timezone(config.MARKET_TIMEZONE)
            now = datetime.now(market_tz)

            if config.PRE_MARKET_FETCH_ENABLED:
                next_opens = []
                if config.TRADE_US_MARKETS:
                    next_opens.append(get_next_market_open('US', now))
                if config.TRADE_EUROPEAN_MARKETS:
                    next_opens.append(get_next_market_open('EUROPEAN', now))
                if config.TRADE_24_HOUR_MARKETS:
                    next_opens.append(get_next_market_open('24_HOUR', now))

                if next_opens:
                    next_market_open = min(next_opens)
                    pre_market_start_time = next_market_open - timedelta(minutes=config.PRE_MARKET_FETCH_MINUTES.get(get_symbol_market_type(config.SYMBOLS[0]), 10))
                    
                    if now < pre_market_start_time:
                        sleep_duration = (pre_market_start_time - now).total_seconds()
                        if sleep_duration > 0:
                            print(f"Sleeping for {sleep_duration:.2f} seconds until pre-market for the next session.")
                            time.sleep(sleep_duration)
                        now = datetime.now(market_tz) # Update time after sleeping

            for symbol in config.SYMBOLS:
                state = states[symbol]
                market_type = get_symbol_market_type(symbol)

                if not market_type:
                    continue

                if now.date() != state['last_checked_day']:
                    state.update({'orh': None, 'orl': None, 'trade_taken_today': False, 'bullish_break': False, 'bearish_break': False, 'last_checked_day': now.date()})

                if state['trade_taken_today']:
                    continue

                if state['orh'] is None:
                    start_of_day = datetime.combine(now.date(), time_obj(0, 0), tzinfo=market_tz)
                    df_today = client.get_data(symbol, config.ORB_TIMEFRAME, start_of_day, datetime.now(pytz.utc))
                    
                    if not df_today.empty:
                        if market_type == 'US':
                            state['orh'], state['orl'] = strategy.get_opening_range(df_today, config.US_MARKET_OPEN_TIME)
                        elif market_type == 'EUROPEAN':
                            state['orh'], state['orl'] = strategy.get_opening_range(df_today, config.EUROPEAN_MARKET_OPEN_TIME)
                        elif market_type == '24_HOUR':
                            state['orh'], state['orl'] = strategy.get_opening_range_24_hour(df_today, now)

                if state['orh'] is not None and not state['trade_taken_today']:
                    live_df = client.get_data_from_pos(symbol, config.ORB_TIMEFRAME, 2)
                    if len(live_df) < 2:
                        continue
                    
                    signals, state['bullish_break'], state['bearish_break'] = strategy.check_trade_signals(symbol, live_df, state['orh'], state['orl'], state['bullish_break'], state['bearish_break'])

                    if signals:
                        for signal in signals:
                            stop_loss_pips = (signal['entry'] - signal['sl']) / mt5.symbol_info(symbol).point if signal['type'] == 'BUY' else (signal['sl'] - signal['entry']) / mt5.symbol_info(symbol).point
                            lot_size = calculate_lot_size(symbol, stop_loss_pips)
                            client.execute_trade(signal['type'], symbol, lot_size, signal['entry'], signal['sl'], signal['tp'], config.MAGIC_NUMBER)
                            state['trade_taken_today'] = True
                            break
            
            # --- Event-Driven Sleep Logic ---
            timeframe_seconds = get_timeframe_in_seconds(config.ORB_TIMEFRAME)
            now_utc = datetime.now(pytz.utc)
            
            # Find the most recent candle time across all symbols
            last_candle_times = []
            for symbol in config.SYMBOLS:
                df = client.get_data_from_pos(symbol, config.ORB_TIMEFRAME, 1)
                if not df.empty:
                    last_candle_times.append(df.index[-1].to_pydatetime())
            
            if last_candle_times:
                last_candle_time = max(last_candle_times)
                next_candle_open = last_candle_time + timedelta(seconds=timeframe_seconds)
                sleep_duration = (next_candle_open - now_utc).total_seconds() + 2

                if sleep_duration > 0:
                    print(f"Sleeping for {sleep_duration:.2f} seconds until the next candle.")
                    time.sleep(sleep_duration)
                else:
                    time.sleep(5)
            else:
                print("Waiting for 60 seconds before next check...")
                time.sleep(60)

        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            client.notifier.send_message(f"*Bot Error*\nAn error occurred in the main loop:\n{escape_markdown_v2(str(e))}")
            time.sleep(60)

def main():
    parser = argparse.ArgumentParser(description="ORB Trading Bot")
    parser.add_argument('--mode', type=str, default='live', choices=['live', 'backtest', 'optimize', 'walk_forward'], help="Execution mode")
    args = parser.parse_args()

    client = None
    try:
        client = BrokerClient()
        client.notifier.send_message(escape_markdown_v2("*ORB Bot Started*\nBot is now running."))
        client.notifier.send_message(f"*Bot Connected*\nAccount: {escape_markdown_v2(str(client.account_login))}")

        if args.mode == 'live':
            run_live(client)
        else:
            print(f"{args.mode.capitalize()} mode should be run from its respective script (e.g., backtest.py)")

    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"A critical error occurred: {e}")
        if client:
            client.notifier.send_message(f"*CRITICAL ERROR*\nBot stopped due to unhandled exception:\n{escape_markdown_v2(str(e))}")
    finally:
        if client:
            client.shutdown()
            client.notifier.send_message(escape_markdown_v2("*Bot Shut Down*\nBot has completed its shutdown sequence."))
        sys.stdout = original_stdout
        log_file.close()

if __name__ == "__main__":
    main()