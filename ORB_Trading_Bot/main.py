# -*- coding: utf-8 -*-
"""
Main entry point for the ORB Trading Bot.
"""

import argparse
import time
from datetime import datetime, time as time_obj
import pytz
from notifier import escape_markdown_v2
import sys
import os

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
output_dir = r'D:\Microsoft VS Code\Projects\2025\ORB_Outputs'
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

def run_live(client):
    """Runs the bot in a continuous loop for live trading."""
    print("\n--- Starting Live Trading Bot ---")
    
    states = {symbol: {'orh': None, 'orl': None, 'last_checked_day': None, 'trade_taken_today': False, 'bullish_break': False, 'bearish_break': False} for symbol in config.SYMBOLS}

    while True:
        try:
            market_tz = pytz.timezone(config.MARKET_TIMEZONE)
            current_market_time = datetime.now(market_tz)
            
            for symbol in config.SYMBOLS:
                state = states[symbol]
                market_type = get_symbol_market_type(symbol)

                if not market_type:
                    continue

                if current_market_time.date() != state['last_checked_day']:
                    state.update({'orh': None, 'orl': None, 'trade_taken_today': False, 'bullish_break': False, 'bearish_break': False, 'last_checked_day': current_market_time.date()})

                if state['trade_taken_today']:
                    continue

                if state['orh'] is None:
                    start_of_day = datetime.combine(current_market_time.date(), time_obj(0, 0), tzinfo=market_tz)
                    df_today = client.get_data(symbol, config.ORB_TIMEFRAME, start_of_day, datetime.now(pytz.utc))
                    
                    if not df_today.empty:
                        if market_type == 'US':
                            state['orh'], state['orl'] = strategy.get_opening_range(df_today, config.US_MARKET_OPEN_TIME)
                        elif market_type == 'EUROPEAN':
                            state['orh'], state['orl'] = strategy.get_opening_range(df_today, config.EUROPEAN_MARKET_OPEN_TIME)
                        elif market_type == '24_HOUR':
                            state['orh'], state['orl'] = strategy.get_opening_range_24_hour(df_today, current_market_time)

                if state['orh'] is not None and not state['trade_taken_today']:
                    live_df = client.get_data_from_pos(symbol, config.ORB_TIMEFRAME, 2)
                    if len(live_df) < 2:
                        continue
                    
                    signals, state['bullish_break'], state['bearish_break'] = strategy.check_trade_signals(live_df, state['orh'], state['orl'], state['bullish_break'], state['bearish_break'])

                    if signals:
                        for signal in signals:
                            client.execute_trade(signal['type'], symbol, config.LOT_SIZE, signal['entry'], signal['sl'], signal['tp'], config.MAGIC_NUMBER)
                            state['trade_taken_today'] = True
                            break

            print("Waiting for next check...")
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