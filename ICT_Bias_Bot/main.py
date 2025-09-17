import time
from datetime import datetime
import pytz
import MetaTrader5 as mt5
import signal
import argparse # New import

from api_client import BrokerClient
import strategy
from config import SYMBOLS, DAILY_OPEN_TIME, DAILY_TIMEFRAME, ENTRY_TIMEFRAMES, NY_TIMEZONE, SL_BUFFER_PIPS, TP_RULE, MAX_SL_PIPS
from notifier import escape_markdown_v2

client = None # Declare client as a global variable

stop_bot_flag = False # Flag to signal bot to stop gracefully

def signal_handler(sig, frame):
    global stop_bot_flag
    print("SIGINT received. Setting stop_bot_flag to True.")
    stop_bot_flag = True
    # Send immediate notification that bot is stopping
    if client is not None and client.notifier.enabled:
        client.notifier.send_message(escape_markdown_v2("*Bot Stopping*\nReceived Ctrl+C. Attempting graceful shutdown."))

def get_daily_open_price(client):
    """Fetches the Daily Open Price based on the config setting (0:00 or 17:00 NYT)."""
    
    if DAILY_OPEN_TIME == "ASIAN":
        daily_data = client.get_data(SYMBOLS[0], DAILY_TIMEFRAME, 2)
        return daily_data.iloc[-1]['Open']
        
    elif DAILY_OPEN_TIME == "MIDNIGHT":
        h1_data = client.get_data(SYMBOLS[0], "H1", 30)
        
        for index, row in h1_data.iterrows():
            dt_ny = index.tz_localize('UTC').astimezone(NY_TIMEZONE)
            if dt_ny.hour == 0 and dt_ny.minute == 0:
                return row['Open']

    return None

def main(args):
    global client # Access the global client variable
    try:
        client = BrokerClient(trade_marker_size=args.marker_size) # Pass marker_size
        print("Broker connection successful. Starting bot...")
        client.notifier.send_message(escape_markdown_v2("*ICT Bias Bot Started*\nBot is now running."))
        client.notifier.send_message(f"*Bot Connected*\nAccount: {escape_markdown_v2(str(client.account_login))}")
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize broker client. {e}")
        # Notification for this case is handled by BrokerClient.__init__
        return

    signal.signal(signal.SIGINT, signal_handler) # Register signal handler

    if args.mode == 'run': # Added mode check
        while not stop_bot_flag: # Loop while flag is False
            try:
                now_utc = datetime.now(pytz.utc)
                ny_time = now_utc.astimezone(NY_TIMEZONE)
                print(f"\n[{now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}] | [{ny_time.strftime('%Y-%m-%d %H:%M:%S %Z')}] Starting new check cycle...")

                for symbol in SYMBOLS:
                    # --- 1. Killer Zone Check ---
                    is_trade_time = strategy.is_in_killer_zone(ny_time, mode='live')
                    if not is_trade_time:
                        print(f"[{symbol}] Currently outside configured Killer Zones. Skipping...")
                        continue
                    print(f"[{symbol}] Within Killer Zone. Proceeding with checks.")

                    # --- 2. Data Fetching ---
                    daily_data = client.get_data(symbol, DAILY_TIMEFRAME, 2)
                    if daily_data.empty or len(daily_data) < 2:
                        print(f"[{symbol}] Could not fetch sufficient daily data. Retrying next cycle.")
                        continue
                    
                    pdh = daily_data.iloc[-2]['High']
                    pdl = daily_data.iloc[-2]['Low']

                    # --- 3. Daily Bias ---
                    daily_bias = strategy.determine_daily_bias(daily_data)
                    print(f"[{symbol}] Daily Bias: {daily_bias}")
                    if 'NO_TRADE' in daily_bias:
                        continue

                    # --- 4. Daily Open ---
                    daily_open_price = get_daily_open_price(client)
                    if daily_open_price is None:
                        print(f"[{symbol}] Could not determine Daily Open price. Retrying next cycle.")
                        continue
                        
                    # --- 5. HTF FVG Check ---
                    for tf in ENTRY_TIMEFRAMES:
                        print(f"[{symbol}] Checking for FVG on {tf}...")
                        tf_data = client.get_data(symbol, tf, 50)
                        if tf_data.empty:
                            print(f"[{symbol}] Could not fetch {tf} data.")
                            continue
                        
                        fvg_info = strategy.check_fvg(tf_data)
                        if not fvg_info:
                            print(f"[{symbol}] No recent FVG found on {tf}.")
                            continue
                        
                        print(f"[{symbol}] Found {fvg_info['type']} FVG on {tf} between {fvg_info['bottom']:.5f} and {fvg_info['top']:.5f}.")

                        # --- 6. FVG Location Check ---
                        if not strategy.check_fvg_location(fvg_info, daily_open_price, daily_bias, mode='live'):
                            print(f"[{symbol}] FVG on {tf} invalidated: Location rule failed.")
                            continue 

                        # --- 7. Price Mitigation Check ---
                        current_tick = mt5.symbol_info_tick(symbol)
                        if not current_tick:
                            print(f"[{symbol}] Could not fetch current tick data.")
                            continue
                        price = current_tick.last
                        if not (fvg_info['bottom'] <= price <= fvg_info['top']):
                            print(f"[{symbol}] Price ({price:.5f}) has not yet mitigated the FVG on {tf}. Skipping.")
                            continue
                        
                        print(f"[{symbol}] Price is currently mitigating the FVG on {tf}.")

                        # --- 8. LTF MSS and Entry ---
                        ltf = "M5" if tf == "H1" else "M15"
                        print(f"[{symbol}] Checking for MSS on {ltf}...")
                        ltf_data = client.get_data(symbol, ltf, 100)
                        if ltf_data.empty:
                            print(f"[{symbol}] Could not fetch {ltf} data.")
                            continue
                        
                        sl_buffer_pips = strategy.get_symbol_specific_setting(SL_BUFFER_PIPS, symbol)
                        pip_size = strategy.get_pip_size(symbol)
                        sl_buffer = sl_buffer_pips * pip_size
                        max_sl_pips = strategy.get_symbol_specific_setting(MAX_SL_PIPS, symbol)

                        entry_details = strategy.check_mss_and_immediate_entry(
                            ltf_data, daily_bias, sl_buffer, pdh, pdl, TP_RULE, symbol, max_sl_pips, pip_size, sl_buffer_pips
                        )
                        
                        if entry_details:
                            print(f"--- VALID ENTRY FOUND on {tf} for {symbol} confirmed by {ltf} MSS ---")
                            
                            sl_price_adjusted = client.adjust_price(
                                symbol, entry_details['sl_price'], is_sl=True
                            )
                            tp_price_adjusted = client.adjust_price(
                                symbol, entry_details['tp_price'], is_sl=False
                            )
                            
                            sl_pips = abs(sl_price_adjusted - entry_details['entry_price']) * client.point_to_pip_factor(symbol)
                            volume = client.calculate_volume(symbol, sl_pips)

                            order_type = 'BUY' if 'LONG' in daily_bias else 'SELL'
                            
                            client.place_order(
                                symbol, order_type, entry_details['entry_price'], 
                                sl_price_adjusted, tp_price_adjusted, volume,
                                fvg_info=fvg_info, tp_rule=entry_details['tp_rule']
                            )
                            break # Exit the inner loop (timeframes) for this symbol
                
                # --- 9. Position Management ---
                client.check_and_manage_positions()
            
            except Exception as e:
                print(f"An error occurred in the main loop: {e}")
                client.notifier.send_message(f"*Bot Error*\nAn error occurred in the main loop:\n{escape_markdown_v2(str(e))}")
                
            # --- 10. Sleep ---
            sleep_duration = 60 # Sleep for 60 seconds
            print(f"\nCycle complete. Sleeping for {sleep_duration} seconds...")
            for _ in range(sleep_duration): # Sleep for 300 seconds, checking flag every second
                if stop_bot_flag:
                    break
                time.sleep(1)

    elif args.mode == 'plot': # New plot mode
        print(f"--- Running in Plotting Mode for symbol: {args.symbol} for last {args.plot_hours} hours ---")
        # IMPORTANT: The plot_symbol_data method in api_client.py needs to be updated
        # to accept and use the 'plot_hours' argument.
        client.plot_symbol_data(args.symbol, plot_hours=args.plot_hours)
        print("Plotting complete.")

    print("Bot stopping gracefully.")
    # Final shutdown message will be sent from the finally block

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICT Bias Bot")
    parser.add_argument(
        '--mode', type=str, default='run', choices=['run', 'plot'],
        help="Specify the bot's mode: 'run' for live trading, 'plot' for generating trade analysis charts."
    )
    parser.add_argument(
        '--symbol', type=str, default='EURUSD', # Default symbol for plotting
        help="The symbol to plot when in 'plot' mode."
    )
    parser.add_argument(
        '--marker_size', type=int, default=25,
        help="Size of the buy/sell markers on the plot. Defaults to 25."
    )
    parser.add_argument(
        '--plot_hours', type=int, default=48,
        help="Number of hours to plot for trade analysis. Defaults to 48."
    )
    args = parser.parse_args()

    try:
        main(args) # Pass args to main
    except SystemExit:
        print("Bot stopped by SystemExit.")
        if client is not None and client.notifier.enabled:
            client.notifier.send_message(escape_markdown_v2("*Bot Stopped*\nBot was stopped by SystemExit."))
    except Exception as e:
        print(f"An unhandled exception occurred outside the main loop: {e}")
        if client is not None and client.notifier.enabled:
            client.notifier.send_message(f"*CRITICAL ERROR*\nBot stopped due to unhandled exception:\n{escape_markdown_v2(str(e))}")
    finally:
        if client is not None:
            if client.notifier.enabled:
                client.notifier.send_message(escape_markdown_v2("*Bot Shut Down*\nBot has completed its shutdown sequence."))
            # mt5.is_initialized() is not a valid function. mt5.shutdown() can be called directly.
            mt5.shutdown()