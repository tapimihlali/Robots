import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os # Added import
import pytz # For datetime.now(pytz.utc)
from datetime import datetime, timedelta # New import

import config
import settings
import strategy # For calculate_atr and get_pip_size
from notifier import Notifier, escape_markdown_v2

# New imports for plotting
import matplotlib.pyplot as plt
import pandas_ta as ta

class BrokerClient:
    def __init__(self, trade_marker_size=25): # Added trade_marker_size parameter
        """Initializes the MT5 connection and Notifier."""
        self.trade_context = {} # Dictionary to store context for each trade
        self.notifier = Notifier()
        MT5_USER, MT5_PASS, MT5_SERVER, MT5_PATH, _, _ = settings.synthetic()
        if not mt5.initialize(path=MT5_PATH, login=MT5_USER, password=MT5_PASS, server=MT5_SERVER):
            print(f"MT5 initialization failed: {mt5.last_error()}")
            self.notifier.send_message(f"🚨 *Bot Error* 🚨\nMT5 initialization failed: {escape_markdown_v2(str(mt5.last_error()))}")
            if not mt5.login(MT5_USER, password=MT5_PASS, server=MT5_SERVER):
                print(f"MT5 login failed: {mt5.last_error()}")
                self.notifier.send_message(f"🚨 *Bot Error* 🚨\nMT5 login failed: {escape_markdown_v2(str(mt5.last_error()))}")
                raise Exception("Broker connection failed.")
        
        account_info = mt5.account_info()
        if account_info is None:
            print("Failed to get account info after MT5 initialization.")
            self.notifier.send_message(f"🚨 *Bot Error* 🚨\nFailed to get account info after MT5 initialization.")
            raise Exception("Failed to get account info.")
        self.account_login = account_info.login

        self.symbol_info = mt5.symbol_info(config.SYMBOLS[0])
        if not self.symbol_info:
             raise Exception(f"Failed to get symbol info for {config.SYMBOLS[0]}.")
        
        self.trade_marker_size = trade_marker_size # Set marker size

    def get_data(self, symbol, timeframe_str, count):
        """Fetches OHLC data and returns a DataFrame."""
        tf_map = {"D1": mt5.TIMEFRAME_D1, "H4": mt5.TIMEFRAME_H4, "H1": mt5.TIMEFRAME_H1, 
                  "M15": mt5.TIMEFRAME_M15, "M5": mt5.TIMEFRAME_M5, "M1": mt5.TIMEFRAME_M1} # Added M1
        timeframe = tf_map.get(timeframe_str)
        if not timeframe:
            raise ValueError(f"Invalid timeframe string: {timeframe_str}")

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            print(f"Error getting rates for {symbol} {timeframe_str}: {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        return df[['open', 'high', 'low', 'close', 'tick_volume']].rename(
            columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}
        )
    
    def get_min_lot_size(self, symbol):
        return self.symbol_info.min_volume

    def get_min_stop_distance(self, symbol):
        return self.symbol_info.trade_stops_level
    
    def point_to_pip_factor(self, symbol):
        return 10**self.symbol_info.digits / 10**4

    def calculate_volume(self, symbol, sl_pips):
        if symbol in config.CUSTOM_LOT_SIZES:
            return config.CUSTOM_LOT_SIZES[symbol]

        account_info = mt5.account_info()
        if account_info is None:
            print("Failed to get account info.")
            return self.get_min_lot_size(symbol)
            
        equity = account_info.equity
        risk_amount = equity * config.RISK_PER_TRADE_PERCENT
        
        calculated_volume = self.get_min_lot_size(symbol) * 5
        
        min_lot = self.get_min_lot_size(symbol)
        return max(calculated_volume, min_lot)

    def adjust_price(self, symbol, price_level, is_sl=True):
        min_distance_points = self.get_min_stop_distance(symbol)
        point = self.symbol_info.point

        current_tick = mt5.symbol_info_tick(symbol)
        if current_tick is None: return price_level

        distance = abs(price_level - current_tick.last) 
        
        if distance < min_distance_points * point:
            adjustment = (min_distance_points + 5) * point
            if is_sl:
                if price_level > current_tick.last:
                    return price_level + adjustment
                else:
                    return price_level - adjustment
        return price_level

    def place_order(self, symbol, order_type, price, sl, tp, volume, fvg_info=None, tp_rule=None):
        if order_type == 'BUY':
            action = mt5.TRADE_ACTION_BUY
            order_type_mt5 = mt5.ORDER_TYPE_BUY
        else: 
            action = mt5.TRADE_ACTION_SELL
            order_type_mt5 = mt5.ORDER_TYPE_SELL

        request = {
            "action": action, "symbol": symbol, "volume": volume, "type": order_type_mt5,
            "price": price, "sl": sl, "tp": tp, "deviation": 20, "magic": config.MAGIC_NUMBER,
            "comment": "ICT_Bias_Bot", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK, 
        }

        result = mt5.order_send(request)
        print(f"Order sent. Result: {result}")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed: {mt5.last_error()}")
            self.notifier.send_message(f"⚠️ *Trade Failed* ⚠️\nSymbol: {escape_markdown_v2(symbol)}\nType: {escape_markdown_v2(order_type)}\nReason: {escape_markdown_v2(str(mt5.last_error()))}")
        else:
            position_ticket = result.order
            self.trade_context[position_ticket] = {
                'fvg': fvg_info, 'initial_sl': sl, 'breakeven_triggered': False 
            }
            print(f"Stored context for order {position_ticket}")
            message = f"✅ *Trade Opened* ✅\nSymbol: {escape_markdown_v2(symbol)}\nType: {escape_markdown_v2(order_type)}\nEntry: {price:.5f}\nSL: {sl:.5f}\nTP: {tp:.5f}\nVolume: {volume}"
            self.notifier.send_message(message)

    def get_open_trades(self, symbol=None):
        print(f"DEBUG: Calling mt5.positions_get with symbol={symbol}") # Added print
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            last_mt5_error = mt5.last_error() # Capture error immediately
            print(f"Failed to retrieve positions: {last_mt5_error}")
            return []
        bot_positions = [p for p in positions if p.magic == config.MAGIC_NUMBER]
        return bot_positions

    def should_close_trade(self, position, ltf_data):
        trade_id = position.ticket
        if trade_id not in self.trade_context: return False
        fvg = self.trade_context[trade_id].get('fvg')
        if not fvg: return False

        if fvg['type'] == 'BULLISH' and ltf_data.iloc[-1]['Close'] < fvg['bottom']:
            print(f"INVALIDATION: Price closed below bullish FVG for ticket {trade_id}.")
            return True
        elif fvg['type'] == 'BEARISH' and ltf_data.iloc[-1]['Close'] > fvg['top']:
            print(f"INVALIDATION: Price closed above bearish FVG for ticket {trade_id}.")
            return True
        return False

    def close_trade(self, position, reason="Manual Close"):
        """Closes a trade and sends a notification."""
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "deviation": 20, "magic": config.MAGIC_NUMBER,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Successfully closed position {position.ticket}")
            message = f"🛑 *Trade Closed* 🛑\nSymbol: {escape_markdown_v2(position.symbol)}\nTicket: {position.ticket}\nReason: {escape_markdown_v2(reason)}\nProfit: {position.profit:.2f}"
            self.notifier.send_message(message)
            if position.ticket in self.trade_context:
                del self.trade_context[position.ticket]
        else:
            print(f"Failed to close position {position.ticket}: {mt5.last_error()}")
            self.notifier.send_message(f"🚨 *Bot Error* 🚨\nFailed to close position {position.ticket}: {escape_markdown_v2(str(mt5.last_error()))}")

    def modify_position(self, ticket, sl, tp):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
            "magic": config.MAGIC_NUMBER,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Successfully modified position {ticket}. New SL: {sl}, New TP: {tp}")
            self.notifier.send_message(f"✅ *Trade Modified* ✅\nTicket: {ticket}\nNew SL: {sl:.5f}\nNew TP: {tp:.5f}")
        else:
            print(f"Failed to modify position {ticket}: {mt5.last_error()}")
            self.notifier.send_message(f"🚨 *Bot Error* 🚨\nFailed to modify position {ticket}: {escape_markdown_v2(str(mt5.last_error()))}")

    def apply_trailing_stop(self, position):
        trade_id = position.ticket
        if trade_id not in self.trade_context:
            return

        context = self.trade_context[trade_id]
        initial_sl = context.get('initial_sl')
        if not initial_sl:
            return

        symbol_info = mt5.symbol_info(position.symbol)
        if not symbol_info:
            return

        current_tick = mt5.symbol_info_tick(position.symbol)
        if not current_tick:
            return

        current_price = current_tick.last
        initial_risk = abs(position.price_open - initial_sl)
        if initial_risk == 0:
            return

        current_profit = (current_price - position.price_open) if position.type == mt5.ORDER_TYPE_BUY else (position.price_open - current_price)
        current_r_multiple = current_profit / initial_risk if initial_risk > 0 else 0

        # Breakeven
        if not context.get('breakeven_triggered', False):
            if current_r_multiple >= config.TRAILING_STOP_LOSS['breakeven_r_multiple']:
                new_sl = position.price_open
                if (position.type == mt5.ORDER_TYPE_BUY and new_sl > position.sl) or \
                   (position.type == mt5.ORDER_TYPE_SELL and new_sl < position.sl):
                    self.modify_position(position.ticket, new_sl, position.tp)
                    self.trade_context[trade_id]['breakeven_triggered'] = True
                    self.notifier.send_message(f"✅ *Trailing Stop* ✅\nSymbol: {escape_markdown_v2(position.symbol)}\nTicket: {position.ticket}\nAction: Moved to Breakeven")

        # ATR Trailing
        if context.get('breakeven_triggered', False):
            atr_data = self.get_data(position.symbol, "M5", config.TRAILING_STOP_LOSS['atr_period'] + 1)
            if len(atr_data) < config.TRAILING_STOP_LOSS['atr_period'] + 1:
                return

            atr_value = strategy.calculate_atr(atr_data['High'], atr_data['Low'], atr_data['Close'], config.TRAILING_STOP_LOSS['atr_period'])
            
            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = current_price - (atr_value * config.TRAILING_STOP_LOSS['atr_multiplier'])
                if new_sl > position.sl:
                    self.modify_position(position.ticket, new_sl, position.tp)
            else: # SELL
                new_sl = current_price + (atr_value * config.TRAILING_STOP_LOSS['atr_multiplier'])
                if new_sl < position.sl:
                    self.modify_position(position.ticket, new_sl, position.tp)

    def check_and_manage_positions(self):
        bot_positions = self.get_open_trades()
        if not bot_positions: return

        for position in bot_positions:
            symbol = position.symbol
            # Get the invalidation setting for the specific symbol, or use the default
            invalidation_setting = config.ENABLE_INVALIDATION_EXIT.get(symbol, config.ENABLE_INVALIDATION_EXIT.get('default', False))

            if invalidation_setting:
                ltf_data = self.get_data(symbol, "M5", 10)
                if self.should_close_trade(position, ltf_data):
                    self.close_trade(position, reason="FVG Invalidation")
                    continue # Skip to next position if this one was closed

            if config.TRAILING_STOP_LOSS['live_enabled']:
                self.apply_trailing_stop(position)

    def _deals_to_dataframe(self, deals):
        """Converts a list of MT5 deals to a pandas DataFrame."""
        if deals is None or not deals:
            return pd.DataFrame()
        try:
            df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            return df
        except (AttributeError, IndexError) as e:
            print(f"Could not convert deals to DataFrame: {e}")
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
            print(f"Could not convert orders to DataFrame: {e}")
            return pd.DataFrame()

    def plot_symbol_data(self, symbol, plot_hours):
        """
        Fetches required data from MT5, plots the chart with trades, and saves it as a PNG file.
        """
        print(f"Preparing detailed plot for {symbol} from MT5 data...")

        # --- Step 1: Fetch price data ---
        num_candles = plot_hours * 60  # plot_hours * 60 minutes/hour
        print(f"Fetching latest {num_candles} M1 candles for {symbol} (approx. {plot_hours} hours).")
        # Use self.get_data which returns a DataFrame with 'Open', 'High', 'Low', 'Close'
        df_entry = self.get_data(symbol, "M1", num_candles)
        if df_entry.empty:
            print(f"No M1 data available for {symbol}. Cannot generate plot.")
            return

        # --- Step 2: Fetch Trade History from MT5 ---
        print(f"Fetching trade history from MT5 for the last {plot_hours} hours.")
        from_date = datetime.now(pytz.utc) - timedelta(hours=plot_hours)
        to_date = datetime.now(pytz.utc)
        print(f"DEBUG: Fetching deals from {from_date} to {to_date} for symbol {symbol}.")
        deals = mt5.history_deals_get(from_date, to_date)
        print(f"DEBUG: Raw deals fetched: {len(deals) if deals is not None else 0}")
        
        trades_to_plot = []
        if deals is None or len(deals) == 0:
            print(f"No trade deals found for any symbol in the last {plot_hours} hours.")
        else:
            deals_df = self._deals_to_dataframe(deals)
            print(f"DEBUG: Deals DataFrame shape: {deals_df.shape}")
            # Filter for the specific symbol, entry deals, and bot's magic number
            entry_deals_df = deals_df[
                (deals_df['symbol'] == symbol) &
                (deals_df['entry'] == mt5.DEAL_ENTRY_IN) &
                (deals_df['magic'] == config.MAGIC_NUMBER) # Use config.MAGIC_NUMBER
            ].copy()
            print(f"DEBUG: Filtered entry deals DataFrame shape: {entry_deals_df.shape}")

            if entry_deals_df.empty:
                print(f"No entry trades found for {symbol} with magic number {config.MAGIC_NUMBER} in the last {plot_hours} hours.")
            else:
                print(f"Found {len(entry_deals_df)} entry trades for {symbol}. Fetching order details...")
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
                            print(f"Could not find order details for ticket {order_ticket}. Plotting without SL/TP.")
                    
                    trades_to_plot.append(trade_info)

        # --- Step 3: Prepare data and plot ---
        df_plot = df_entry.copy()
        # Rename columns to be compatible with pandas_ta if not already
        df_plot = df_plot.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})

        df_plot['ema_entry'] = ta.ema(df_plot['close'], length=config.EMA_ENTRY_PERIOD)
        df_plot['ema_trend'] = ta.ema(df_plot['close'], length=config.EMA_TREND_PERIOD)

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(22, 10))
        ax.plot(df_plot.index, df_plot['close'], label="Price (M1)", color="blue", alpha=0.8, linewidth=1)
        ax.plot(df_plot.index, df_plot['ema_entry'], label=f'EMA {config.EMA_ENTRY_PERIOD} (M1)', color='orange', linestyle='--',
                linewidth=1.5)
        ax.plot(df_plot.index, df_plot['ema_trend'], label=f'EMA {config.EMA_TREND_PERIOD} (M1)', color='gray', linestyle='--',
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
        plot_filename_base = f"{symbol.replace(' ', '_').replace('(', '').replace(')', '')}_trade_analysis"
        
        # Define the directory where plots will be saved. Assuming project root for now.
        # You might want to make this configurable or create a 'plots' subdirectory.
        plot_dir = config.OUTPUT_PATH # Use the configured output path
        os.makedirs(plot_dir, exist_ok=True) # Ensure the directory exists

        # Generate a filename and save the plot, overwriting if it exists.
        plot_filename = f"{plot_filename_base}.png"
        full_plot_path = os.path.join(plot_dir, plot_filename)
        
        fig.savefig(full_plot_path)
        plt.close(fig)
        print(f"Chart has been saved as {full_plot_path}. You can open this file in VS Code to view it.")
        pass