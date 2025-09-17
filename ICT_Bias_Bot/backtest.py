import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
from contextlib import redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt
from scipy import stats as st
import argparse
import os

# Import bot components
import strategy
import config
import settings

INITIAL_EQUITY = 100.0
DEFAULT_BACKTEST_DAYS = 180

# --- BACKTEST CONFIGURATION ---

def _calculate_consecutive_streaks(series):
    """Helper to calculate max consecutive wins and losses."""
    wins = (series > 0).astype(int)
    losses = (series <= 0).astype(int)
    
    max_wins = 0
    current_wins = 0
    for w in wins:
        if w == 1:
            current_wins += 1
        else:
            max_wins = max(max_wins, current_wins)
            current_wins = 0
    max_wins = max(max_wins, current_wins)

    max_losses = 0
    current_losses = 0
    for l in losses:
        if l == 1:
            current_losses += 1
        else:
            max_losses = max(max_losses, current_losses)
            current_losses = 0
    max_losses = max(max_losses, current_losses)
    
    return max_wins, max_losses

import os # Ensure os is imported
import config # Ensure config is imported

def plot_pnl_histogram(results_df, symbol):
    """Generates and saves a histogram of profit/loss distribution."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['profit'], bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Profit/Loss Distribution for {symbol}')
    plt.xlabel('Profit/Loss ($)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    output_dir = config.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'{symbol}_pnl_histogram.png')
    
    plt.savefig(file_path)
    plt.close()
    print(f"Generated P/L histogram for {symbol}: {file_path}")

def plot_trades_on_price_chart(price_data, trades_df, symbol):
    """
    Plots price, EMAs, and simulated trades on a single chart.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(22, 10))

    # Ensure price_data has 'Close' column for EMA calculation
    if 'Close' not in price_data.columns:
        print("Error: 'Close' column not found in price_data for plotting.")
        return

    # Plot price data
    ax.plot(price_data.index, price_data['Close'], label='Price (M5)', color='lightgray', linewidth=1)

    # Calculate and plot EMAs
    ema_entry = price_data['Close'].ewm(span=config.EMA_ENTRY_PERIOD, adjust=False).mean()
    ema_trend = price_data['Close'].ewm(span=config.EMA_TREND_PERIOD, adjust=False).mean()
    ax.plot(price_data.index, ema_entry, label=f'EMA {config.EMA_ENTRY_PERIOD}', color='blue', linewidth=0.8)
    ax.plot(price_data.index, ema_trend, label=f'EMA {config.EMA_TREND_PERIOD}', color='purple', linewidth=0.8)

    # Plot trades
    for _, trade in trades_df.iterrows():
        open_time = pd.to_datetime(trade['open_time']).tz_localize('UTC')
        close_time = pd.to_datetime(trade['close_time']).tz_localize('UTC')
        entry_price = trade['entry_price']
        sl_price = trade['sl']
        tp_price = trade['tp']
        trade_type = trade['type']

        color = 'green' if trade_type == 'BUY' else 'red'
        marker = '^' if trade_type == 'BUY' else 'v'

        # Plot entry marker
        ax.plot(open_time, entry_price, marker=marker, color=color, markersize=8, label=f'{trade_type} Entry')

        # Plot SL and TP lines
        ax.hlines(sl_price, open_time, close_time, color='red', linestyle='--', linewidth=1, label='Stop Loss')
        ax.hlines(tp_price, open_time, close_time, color='green', linestyle='--', linewidth=1, label='Take Profit')

    # Final Plot Configuration
    ax.set_title(f"Backtest Trade Analysis for {symbol}", fontsize=18)
    ax.set_xlabel("Time (UTC)", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    
    # Handle duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10)
    
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    fig.tight_layout()

    # Save plot as PNG
    output_dir = config.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f"{symbol.replace(' ', '_').replace('(', '').replace(')', '')}_backtest_Plot_chart.png")
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Backtest chart saved as {plot_filename}.")



def plot_results(equity_df, balance_df, symbol):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
    fig.suptitle(f'Backtest Results: {symbol}', fontsize=16)

    ax1.plot(equity_df.index, equity_df['equity'], label='Equity', color='blue')
    ax1.plot(balance_df.index, balance_df['balance'], label='Balance', color='orange', linestyle='--')
    ax1.set_ylabel('Value ($)')
    ax1.set_title('Equity & Balance Curve')
    ax1.legend()

    peak_equity = equity_df['equity'].cummax()
    drawdown_val = equity_df['equity'] - peak_equity
    drawdown_pct = (drawdown_val / peak_equity) * 100
    ax2.fill_between(drawdown_pct.index, drawdown_pct, 0, color='red', alpha=0.3, label='Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Equity Drawdown')
    ax2.legend()

    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_dir = config.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'{symbol}_backtest_chart.png')
    plt.savefig(file_path)
    print(f"Generated backtest chart for {symbol}")

def calculate_performance_metrics(trades_df, equity_curve, balance_curve, initial_equity, max_sl_invalidations_count=0):
    total_trades = len(trades_df)
    if total_trades == 0:
        return {}

    equity_df = pd.DataFrame(equity_curve, columns=['time', 'equity'])
    equity_df['time'] = pd.to_datetime(equity_df['time']).dt.tz_localize(None)
    equity_df = equity_df.set_index('time')
    
    wins = trades_df[trades_df['profit'] > 0]
    losses = trades_df[trades_df['profit'] <= 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0

    gross_profit = wins['profit'].sum()
    gross_loss = abs(losses['profit'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = wins['profit'].mean() if not wins.empty else 0
    avg_loss = abs(losses['profit'].mean()) if not losses.empty else 0
    
    total_profit = trades_df['profit'].sum()
    
    daily_returns = equity_df['equity'].resample('D').last().pct_change(fill_method=None).dropna()
    
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = (daily_returns.mean() * 252) / (downside_returns.std() * np.sqrt(252)) if downside_returns.std() > 0 else float('inf')

    peak_equity = equity_df['equity'].cummax()
    drawdown = (equity_df['equity'] - peak_equity)
    max_drawdown_pct = (drawdown / peak_equity).min() * 100
    max_drawdown_dollars = drawdown.min()

    annual_return = daily_returns.mean() * 252
    calmar_ratio = annual_return / abs(max_drawdown_pct) if max_drawdown_pct != 0 else float('inf')

    trades_df['holding_time'] = pd.to_datetime(trades_df['close_time']) - pd.to_datetime(trades_df['open_time'])
    avg_holding_time = trades_df['holding_time'].mean()
    max_consecutive_wins, max_consecutive_losses = _calculate_consecutive_streaks(trades_df['profit'])

    sl_hits = len(trades_df[trades_df['reason'] == 'SL Hit'])
    tp_hits = len(trades_df[trades_df['reason'] == 'TP Hit'])
    invalidation_closes = len(trades_df[trades_df['reason'] == 'Invalidation'])

    biggest_profit = wins['profit'].max() if not wins.empty else 0
    biggest_loss = losses['profit'].min() if not losses.empty else 0

    return {
        'total_trades': total_trades,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown_pct,
        'max_drawdown_dollars': max_drawdown_dollars,
        'sharpe_ratio': sharpe_ratio,
        'pnl': total_profit,
        'equity_curve': equity_curve,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'avg_holding_time': str(avg_holding_time),
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'sl_hits': sl_hits,
        'tp_hits': tp_hits,
        'invalidation_closes': invalidation_closes,
        'biggest_profit': biggest_profit,
        'biggest_loss': biggest_loss,
        'max_sl_invalidations': max_sl_invalidations_count
    }

def print_results(client, plot_title, generate_plot=True, start_date=None, end_date=None, trailing_stop_enabled=None, invalidation_exit_enabled=None):
    report_generation_time = datetime.now()
    print("--- BACKTEST COMPLETE ---")
    start_date_str = start_date.strftime('%Y-%m-%d') if start_date else 'N/A'
    end_date_str = end_date.strftime('%Y-%m-%d') if end_date else 'N/A'
    
    test_duration_days = 'N/A'
    if start_date and end_date:
        test_duration_days = (end_date - start_date).days

    print(f"Period: {start_date_str} to {end_date_str} ({test_duration_days} days)")
    print(f"Trailing Stoploss: {'On' if trailing_stop_enabled else 'Off'}")
    print(f"Invalidation Exit: {'On' if invalidation_exit_enabled else 'Off'}")
    print(f"Report Generated: {report_generation_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbols Tested: {client.symbols}")
    print(f"Initial Equity: ${INITIAL_EQUITY:.2f}")
    print(f"Final Equity: ${client.equity:.2f}")
    total_profit = client.equity - INITIAL_EQUITY
    print(f"Total Net Profit: ${total_profit:.2f}")

    total_trades = len(client.trade_history)
    if total_trades == 0:
        print("No trades were executed.")
        return [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0

    results_df = pd.DataFrame(client.trade_history)
    if 'open_time' in results_df.columns:
        results_df['open_time'] = results_df['open_time'].dt.tz_localize(None)
    if 'close_time' in results_df.columns:
        results_df['close_time'] = results_df['close_time'].dt.tz_localize(None)

    equity_df = pd.DataFrame(client.equity_curve, columns=['time', 'equity']).set_index('time')
    balance_df = pd.DataFrame(client.balance_curve, columns=['time', 'balance']).set_index('time')

    performance = calculate_performance_metrics(results_df, client.equity_curve, client.balance_curve, INITIAL_EQUITY, client.max_sl_invalidations)

    verdict = ""
    if performance.get('profit_factor', 0) > 1.5 and performance.get('win_rate', 0) > 50 and performance.get('sharpe_ratio', 0) > 1.0:
        verdict = "Excellent: Profitable with strong risk-adjusted returns and consistency."
    elif performance.get('profit_factor', 0) > 1.0 and performance.get('win_rate', 0) > 40:
        verdict = "Promising: Shows profitability, but could be improved."
    elif performance.get('pnl', 0) > 0:
        verdict = "Needs Review: Profitable but with significant drawbacks (e.g., high drawdown)."
    else:
        verdict = "Avoid: Unprofitable strategy."

    summary_data = {
        'Metric': [
            'Initial Equity',
            'Risk Assessment (Max Drawdown)', 'Profitability (Net PnL)', 'Profitability (Profit Factor)',
            'Consistency (Sharpe Ratio)', 'Consistency (Win Rate %)', 'Advanced (Sortino Ratio)', 'Advanced (Calmar Ratio)',
            'Total Trades', 'Average Win', 'Average Loss', 'Avg. Holding Time', 'Max Consecutive Wins', 'Max Consecutive Losses', 'Trades Invalidated (Max SL)', 'Balanced Verdict'
        ],
        'Value': [
            f"${INITIAL_EQUITY:.2f}",
            f"{performance.get('max_drawdown_pct', 0):.2f}%", f"${performance.get('pnl', 0):.2f}", f"{performance.get('profit_factor', 0):.2f}",
            f"{performance.get('sharpe_ratio', 0):.2f}", f"{performance.get('win_rate', 0):.2f}%", f"{performance.get('sortino_ratio', 0):.2f}", f"{performance.get('calmar_ratio', 0):.2f}",
            performance.get('total_trades', 0), f"${performance.get('avg_win', 0):.2f}", f"${performance.get('avg_loss', 0):.2f}", performance.get('avg_holding_time', 'N/A'),
            performance.get('max_consecutive_wins', 0), performance.get('max_consecutive_losses', 0), performance.get('max_sl_invalidations', 0), verdict
        ]
    }
    summary_report_df = pd.DataFrame(summary_data)

    results_for_optimizer = {
        'Symbol': plot_title,
        'Initial Equity': INITIAL_EQUITY,
        'Total Trades': performance.get('total_trades', 0),
        'Win Rate (%)': performance.get('win_rate', 0),
        'Profit Factor': performance.get('profit_factor', 0),
        'Net Profit': performance.get('pnl', 0),
        'Sharpe Ratio': performance.get('sharpe_ratio', 0),
        'Sortino Ratio': performance.get('sortino_ratio', 0),
        'Calmar Ratio': performance.get('calmar_ratio', 0),
        'Max Drawdown (%)': performance.get('max_drawdown_pct', 0),
        'Avg. Holding Time': performance.get('avg_holding_time', 'N/A'),
        'Max Consecutive Wins': performance.get('max_consecutive_wins', 0),
        'Max Consecutive Losses': performance.get('max_consecutive_losses', 0),
        'Trades Invalidated (Max SL)': performance.get('max_sl_invalidations', 0),
    }

    excel_results_df = results_df.copy()
    if 'open_time' in excel_results_df.columns:
        excel_results_df['open_time'] = pd.to_datetime(excel_results_df['open_time']).dt.tz_localize(None)
    if 'close_time' in excel_results_df.columns:
        excel_results_df['close_time'] = pd.to_datetime(excel_results_df['close_time']).dt.tz_localize(None)

    output_dir = config.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    excel_file_path = os.path.join(output_dir, f'ICT_backtest_results_{plot_title}.xlsx')
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        summary_report_df.to_excel(writer, sheet_name='Performance Summary', index=False)
        excel_results_df.to_excel(writer, sheet_name='Trade History', index=False)

    print(f"Performance summary and trade history saved to ICT_backtest_results_{plot_title}.xlsx")

    if generate_plot:
        plot_results(equity_df, balance_df, plot_title)
        plot_pnl_histogram(results_df, plot_title)
        if client.symbols:
            price_data = client.all_data[client.symbols[0]].get('M5')
            if price_data is not None and not price_data.empty:
                plot_trades_on_price_chart(price_data, results_df, plot_title)
        
    return [results_for_optimizer], results_df, equity_df, balance_df, client.max_sl_invalidations


class BacktestClient:
    def __init__(self, symbols, start_date, end_date, param_overrides={}, invalidation_exit_enabled=False):
        print("Initializing Backtest Client...")
        self.symbols = symbols
        self.balance = INITIAL_EQUITY
        self.equity = INITIAL_EQUITY
        self.open_positions = []
        self.trade_history = []
        self.trade_context = {}
        self.equity_curve = [(start_date, INITIAL_EQUITY)]
        self.balance_curve = [(start_date, INITIAL_EQUITY)]
        self.param_overrides = param_overrides
        self.invalidation_exit_enabled = invalidation_exit_enabled
        self.max_sl_invalidations = 0

        self._connect_to_mt5()
        self.symbol_infos = self._get_symbol_infos()
        self.symbols = list(self.symbol_infos.keys())
        self.all_data = self._download_all_data(start_date, end_date)
        self.precalculated_daily_bias = self._precalculate_daily_bias()
        print("Historical data download complete.")
        mt5.shutdown()
        print("MT5 connection closed.")

    def _connect_to_mt5(self):
        MT5_USER, MT5_PASS, MT5_SERVER, MT5_PATH, _, _ = settings.synthetic()
        if not mt5.initialize(path=MT5_PATH, login=MT5_USER, password=MT5_PASS, server=MT5_SERVER):
            raise Exception(f"MT5 initialization failed: {mt5.last_error()}")

    def _get_symbol_infos(self):
        infos = {}
        for symbol in self.symbols:
            info = mt5.symbol_info(symbol)
            if info:
                infos[symbol] = info
            else:
                print(f"Warning: Could not get info for symbol '{symbol}'. It will be skipped.")
        return infos

    def _download_all_data(self, start_date, end_date):
        all_data = {symbol: {} for symbol in self.symbols}
        timeframes = {"D1": mt5.TIMEFRAME_D1, "H4": mt5.TIMEFRAME_H4, "H1": mt5.TIMEFRAME_H1, 
                      "M15": mt5.TIMEFRAME_M15, "M5": mt5.TIMEFRAME_M5}
        
        latest_start_date = start_date
        symbols_to_run = self.symbols[:]
        for symbol in symbols_to_run:
            m5_rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
            if m5_rates is None or len(m5_rates) == 0:
                print(f"Warning: No M5 data for {symbol} in range. It will be skipped. MT5 Error: {mt5.last_error()}")
                self.symbols.remove(symbol)
                continue
            
            first_candle_time = pd.to_datetime(m5_rates[0]['time'], unit='s').tz_localize('UTC')
            if first_candle_time > latest_start_date:
                latest_start_date = first_candle_time
        if latest_start_date:
            print(f"Backtest will run from {latest_start_date.strftime('%Y-%m-%d')} to ensure all symbols have data.")
        else:
            print("Error: Could not determine the latest start date for the backtest.")

        for symbol in self.symbols:
            print(f"--- Downloading data for {symbol} ---")
            for tf_str, tf_mt5 in timeframes.items():
                rates = mt5.copy_rates_range(symbol, tf_mt5, latest_start_date, end_date)
                if rates is None or len(rates) == 0:
                    print(f"Warning: No {tf_str} data for {symbol}.")
                    empty_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
                    empty_df.index = pd.to_datetime(empty_df.index)
                    empty_df.index.name = 'time'
                    all_data[symbol][tf_str] = empty_df
                    continue
                
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                df.index = df.index.tz_localize('UTC')
                df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
                all_data[symbol][tf_str] = df
        return all_data

    def _precalculate_daily_bias(self):
        print("Pre-calculating daily biases...")
        daily_bias_data = {symbol: {} for symbol in self.symbols}
        for symbol in self.symbols:
            daily_df = self.all_data.get(symbol, {}).get(config.DAILY_TIMEFRAME)
            if daily_df is None or daily_df.empty:
                continue
            
            daily_df = daily_df.sort_index()

            for i in range(1, len(daily_df)):
                current_day_data = daily_df.iloc[i-1:i+1]
                current_date = current_day_data.index[-1].date()
                
                bias = strategy.determine_daily_bias(current_day_data)
                pdh = current_day_data.iloc[-2]['High']
                pdl = current_day_data.iloc[-2]['Low']
                daily_bias_data[symbol][current_date] = {'bias': bias, 'pdh': pdh, 'pdl': pdl}
        print("Daily bias pre-calculation complete.")
        return daily_bias_data

    def get_data(self, symbol, timeframe_str, count, current_time):
        df = self.all_data.get(symbol, {}).get(timeframe_str)
        if df is None or df.empty:
            return pd.DataFrame()
        end_idx = df.index.searchsorted(current_time, side='right')
        start_idx = max(0, end_idx - count)
        return df.iloc[start_idx:end_idx]

    def get_current_price(self, symbol, current_time):
        m5_data = self.get_data(symbol, "M5", 1, current_time)
        if not m5_data.empty and m5_data.index[0] == current_time:
            return m5_data.iloc[-1]['Close']
        return None

    def has_open_position(self, symbol):
        for pos in self.open_positions:
            if pos['symbol'] == symbol:
                return True
        return False

    def place_order(self, symbol, order_type, price, sl, tp, volume, fvg_info, current_time, tp_rule):
        trade_id = len(self.trade_history) + len(self.open_positions) + 1
        position = {
            'ticket': trade_id, 'symbol': symbol, 'type': order_type,
            'entry_price': price, 'sl': sl, 'tp': tp, 'volume': volume,
            'open_time': current_time, 'profit': 0, 'tp_rule': tp_rule,
            'initial_sl': sl, 'breakeven_triggered': False
        }
        self.open_positions.append(position)
        self.trade_context[trade_id] = {'fvg': fvg_info}
        print(f"--- TRADE OPENED ({symbol}) ---")
        print(f"Time: {current_time}, Type: {order_type}, Entry: {price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}, TP Rule: {tp_rule}")

    def check_and_manage_positions(self, current_time):
        positions_to_close = []
        for pos in self.open_positions:
            symbol = pos['symbol']
            candle = self.get_data(symbol, "M5", 1, current_time)
            if candle.empty or candle.index[0] != current_time: continue

            candle_low, candle_high, candle_close = candle.iloc[-1]['Low'], candle.iloc[-1]['High'], candle.iloc[-1]['Close']

            if pos['type'] == 'BUY' and candle_low <= pos['sl']:
                self._close_position(pos, pos['sl'], current_time, "SL Hit")
                positions_to_close.append(pos)
            elif pos['type'] == 'BUY' and candle_high >= pos['tp']:
                self._close_position(pos, pos['tp'], current_time, "TP Hit")
                positions_to_close.append(pos)
            elif pos['type'] == 'SELL' and candle_high >= pos['sl']:
                self._close_position(pos, pos['sl'], current_time, "SL Hit")
                positions_to_close.append(pos)
            elif pos['type'] == 'SELL' and candle_low <= pos['tp']:
                self._close_position(pos, pos['tp'], current_time, "TP Hit")
                positions_to_close.append(pos)
            else:
                invalidation_setting = self.invalidation_exit_enabled.get(symbol, self.invalidation_exit_enabled.get('default', False))
                if invalidation_setting:
                    ltf_data = self.get_data(symbol, "M5", 10, current_time)
                    if self._should_close_trade(pos, ltf_data):
                        self._close_position(pos, candle_close, current_time, "Invalidation")
                        positions_to_close.append(pos)
                        continue

                trailing_stop_enabled = self.param_overrides.get('TRAILING_STOP_LOSS_ENABLED', config.TRAILING_STOP_LOSS['backtest_enabled'])
                if trailing_stop_enabled:
                    self.apply_trailing_stop(pos, candle_close, current_time)

        self.open_positions = [p for p in self.open_positions if p not in positions_to_close]
        
        floating_pl = 0
        for pos in self.open_positions:
            current_price = self.get_current_price(pos['symbol'], current_time)
            if current_price:
                point = self.symbol_infos[pos['symbol']].point
                pips = (current_price - pos['entry_price']) / point if pos['type'] == 'BUY' else (pos['entry_price'] - current_price) / point
                floating_pl += pips * self.symbol_infos[pos['symbol']].trade_tick_value * pos['volume']
        
        self.equity = self.balance + floating_pl
        self.equity_curve.append((current_time, self.equity))

    def apply_trailing_stop(self, pos, current_price, current_time):
        initial_risk = abs(pos['entry_price'] - pos['initial_sl'])
        if initial_risk == 0: return

        current_profit = (current_price - pos['entry_price']) if pos['type'] == 'BUY' else (pos['entry_price'] - current_price)
        current_r_multiple = current_profit / initial_risk if initial_risk > 0 else 0

        if not pos.get('breakeven_triggered', False):
            if current_r_multiple >= config.TRAILING_STOP_LOSS['breakeven_r_multiple']:
                if pos['type'] == 'BUY' and pos['entry_price'] > pos['sl']:
                    pos['sl'] = pos['entry_price']
                    pos['breakeven_triggered'] = True
                    print(f"--- TRAILING STOP (BREAKEVEN) --- Ticket: {pos['ticket']}, New SL: {pos['sl']:.5f}")
                elif pos['type'] == 'SELL' and pos['entry_price'] < pos['sl']:
                    pos['sl'] = pos['entry_price']
                    pos['breakeven_triggered'] = True
                    print(f"--- TRAILING STOP (BREAKEVEN) --- Ticket: {pos['ticket']}, New SL: {pos['sl']:.5f}")

        if pos.get('breakeven_triggered', False):
            atr_data = self.get_data(pos['symbol'], "M5", config.TRAILING_STOP_LOSS['atr_period'] + 1, current_time)
            if len(atr_data) < config.TRAILING_STOP_LOSS['atr_period'] + 1: return

            atr_value = strategy.calculate_atr(atr_data['High'], atr_data['Low'], atr_data['Close'], config.TRAILING_STOP_LOSS['atr_period'])
            
            if pos['type'] == 'BUY':
                new_sl = current_price - (atr_value * config.TRAILING_STOP_LOSS['atr_multiplier'])
                if new_sl > pos['sl']:
                    pos['sl'] = new_sl
                    print(f"--- TRAILING STOP (ATR) --- Ticket: {pos['ticket']}, New SL: {pos['sl']:.5f}")
            else: # SELL
                new_sl = current_price + (atr_value * config.TRAILING_STOP_LOSS['atr_multiplier'])
                if new_sl < pos['sl']:
                    pos['sl'] = new_sl
                    print(f"--- TRAILING STOP (ATR) --- Ticket: {pos['ticket']}, New SL: {pos['sl']:.5f}")

    def _should_close_trade(self, position, ltf_data):
        fvg = self.trade_context[position['ticket']].get('fvg')
        if not fvg or ltf_data.empty: return False
        if fvg['type'] == 'BULLISH' and ltf_data.iloc[-1]['Close'] < fvg['bottom']: return True
        if fvg['type'] == 'BEARISH' and ltf_data.iloc[-1]['Close'] > fvg['top']: return True
        return False

    def _close_position(self, pos, close_price, close_time, reason):
        point = self.symbol_infos[pos['symbol']].point
        pips = (close_price - pos['entry_price']) / point if pos['type'] == 'BUY' else (pos['entry_price'] - close_price) / point
        profit = pips * self.symbol_infos[pos['symbol']].trade_tick_value * pos['volume']
        
        pos.update({
            'close_price': close_price, 
            'close_time': close_time, 
            'profit': profit, 
            'reason': reason
        })
        self.balance += profit
        self.trade_history.append(pos)
        self.balance_curve.append((close_time, self.balance))
        print(f"--- TRADE CLOSED ({pos['symbol']}) --- Ticket: {pos['ticket']}, Reason: {reason}, Profit: {profit:.2f}")

    def calculate_volume(self, symbol, sl_distance_in_price_points):
        custom_lot_sizes = self.param_overrides.get('CUSTOM_LOT_SIZES', config.CUSTOM_LOT_SIZES)
        if symbol in custom_lot_sizes:
            return custom_lot_sizes[symbol]

        symbol_info = self.symbol_infos.get(symbol)
        if not symbol_info:
            print(f"Warning: Could not get symbol info for {symbol} to calculate volume.")
            return 0.0

        default_lot_strategy = self.param_overrides.get('DEFAULT_LOT_SIZE_STRATEGY', config.DEFAULT_LOT_SIZE_STRATEGY)
        if default_lot_strategy == "MIN_LOT":
            return symbol_info.volume_min
        elif default_lot_strategy == "RISK_PERCENT":
            risk_percent = self.param_overrides.get('RISK_PER_TRADE_PERCENT', config.RISK_PER_TRADE_PERCENT)
            point = symbol_info.point
            tick_value = symbol_info.trade_tick_value
            tick_size = symbol_info.trade_tick_size

            if tick_size == 0 or point == 0:
                return symbol_info.volume_min

            value_per_unit_of_volume = tick_value / tick_size
            risk_amount = self.equity * risk_percent

            if sl_distance_in_price_points == 0:
                return symbol_info.volume_min

            lot_size = risk_amount / (sl_distance_in_price_points * value_per_unit_of_volume)
            lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
            if symbol_info.volume_step > 0:
                lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
            
            return round(lot_size, 2)

        return symbol_info.volume_min

def run_backtest(symbol_to_test=None, param_overrides={}, generate_plot=True, start_date=None, end_date=None, trailing_stop_enabled=None, invalidation_exit_enabled=None, mode='backtest'):
    symbols = [symbol_to_test] if symbol_to_test else config.SYMBOLS
    client = BacktestClient(symbols, start_date, end_date, param_overrides, invalidation_exit_enabled=invalidation_exit_enabled)
    
    all_m5_indices = [client.all_data[s]['M5'].index for s in client.symbols if s in client.all_data and not client.all_data[s]['M5'].empty]
    if not all_m5_indices:
        print("No M5 data available for any symbol. Cannot run backtest.")
        return print_results(client, symbol_to_test or "Portfolio", generate_plot=generate_plot, start_date=start_date, end_date=end_date, trailing_stop_enabled=trailing_stop_enabled, invalidation_exit_enabled=invalidation_exit_enabled)
    
    master_timeline = pd.Index([])
    for idx in all_m5_indices:
        master_timeline = master_timeline.union(idx)
    
    print(f"Starting backtest from {master_timeline[0]} to {master_timeline[-1]}...")
    for current_time in master_timeline:
        client.check_and_manage_positions(current_time)

        for symbol in client.symbols:
            if client.get_current_price(symbol, current_time) is None: continue

            max_open_trades_global = client.param_overrides.get('MAX_OPEN_TRADES_GLOBAL', config.MAX_OPEN_TRADES_GLOBAL)
            custom_max_open_trades = client.param_overrides.get('CUSTOM_MAX_OPEN_TRADES', config.CUSTOM_MAX_OPEN_TRADES)
            max_open_trades_for_symbol = custom_max_open_trades.get(symbol, max_open_trades_global)
            if len([pos for pos in client.open_positions if pos['symbol'] == symbol]) >= max_open_trades_for_symbol:
                continue

            current_day = current_time.date()
            if current_day in client.precalculated_daily_bias[symbol]:
                bias_info = client.precalculated_daily_bias[symbol][current_day]
                bias = bias_info['bias']
                pdh = bias_info['pdh']
                pdl = bias_info['pdl']
            else:
                daily_data = client.get_data(symbol, config.DAILY_TIMEFRAME, 2, current_time)
                if len(daily_data) < 2: continue
                pdh = daily_data.iloc[-2]['High']
                pdl = daily_data.iloc[-2]['Low']
                bias = strategy.determine_daily_bias(daily_data)
            
            if 'NO_TRADE' in bias: continue

            daily_open_price_data = client.get_data(symbol, config.DAILY_TIMEFRAME, 1, current_time)
            if daily_open_price_data.empty: continue
            daily_open_price = daily_open_price_data.iloc[-1]['Open']

            for tf in config.ENTRY_TIMEFRAMES:
                tf_data = client.get_data(symbol, tf, 50, current_time)
                if tf_data.empty: continue
                
                fvg_info = strategy.check_fvg(tf_data)
                if not fvg_info: continue

                if not strategy.check_fvg_location(fvg_info, daily_open_price, bias, mode=mode):
                    continue

                price = client.get_current_price(symbol, current_time)
                if not price or not (fvg_info['bottom'] <= price <= fvg_info['top']): continue
                
                ltf = "M5" if tf == "H1" else "M15"
                ltf_data = client.get_data(symbol, ltf, 100, current_time)
                
                sl_buffer_pips = strategy.get_symbol_specific_setting(client.param_overrides.get('SL_BUFFER_PIPS', config.SL_BUFFER_PIPS), symbol)
                pip_size = strategy.get_pip_size(symbol)
                
                custom_risk_params_for_symbol = strategy.get_symbol_specific_setting(
                    client.param_overrides.get('CUSTOM_RISK_PARAMETERS', config.CUSTOM_RISK_PARAMETERS), 
                    symbol
                )

                if custom_risk_params_for_symbol:
                    tp_rule_for_symbol = custom_risk_params_for_symbol.get('tp_rule', strategy.get_symbol_specific_setting(client.param_overrides.get('TP_RULE', config.TP_RULE), symbol))
                    sl_multiplier = custom_risk_params_for_symbol.get('sl_multiplier', 1.0)
                else:
                    tp_rule_for_symbol = strategy.get_symbol_specific_setting(client.param_overrides.get('TP_RULE', config.TP_RULE), symbol)
                    sl_multiplier = 1.0

                calculated_sl_buffer = sl_buffer_pips * sl_multiplier * pip_size

                max_sl_pips = strategy.get_symbol_specific_setting(client.param_overrides.get('MAX_SL_PIPS', config.MAX_SL_PIPS), symbol)

                trade_decision_result = strategy.get_trade_decision(symbol, tf_data, ltf_data, bias, daily_open_price, pdh, pdl, mode, tp_rule_for_symbol, sl_buffer_pips, max_sl_pips)
                
                if trade_decision_result == "MAX_SL_EXCEEDED":
                    client.max_sl_invalidations += 1
                    continue # Move to the next timeframe or symbol
                elif trade_decision_result: # If it's a valid entry_details dictionary
                    entry_details = trade_decision_result
                    if not strategy.is_in_killer_zone(current_time.to_pydatetime(), mode=mode): continue
                    
                    print(f"[{current_time}] TRADE SIGNAL: {symbol} | {bias} on {tf} -> {ltf}")
                    sl_distance = abs(entry_details['entry_price'] - entry_details['sl_price'])
                    client.place_order(symbol, 'BUY' if 'LONG' in bias else 'SELL', entry_details['entry_price'], entry_details['sl_price'], entry_details['tp_price'], client.calculate_volume(symbol, sl_distance), fvg_info, current_time, entry_details['tp_rule'])
                    break
    
    return print_results(client, symbol_to_test or "Portfolio", generate_plot=generate_plot, start_date=start_date, end_date=end_date, trailing_stop_enabled=trailing_stop_enabled, invalidation_exit_enabled=invalidation_exit_enabled)


def _write_backtest_summary_sheet(writer, performance, initial_capital):
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
    summary_sheet.write('A5', f"The maximum drawdown was {drawdown_pct:.2f}%. This is considered a {risk_level} level of risk.")

    summary_sheet.write('A7', '2. Profitability Confirmation', bold)
    profitability = "Unprofitable"
    if profit_factor > 1.5:
        profitability = "Strong"
    elif profit_factor > 1:
        profitability = "Modest"
    summary_sheet.write('A8', f"The strategy was {'profitable' if pnl > 0 else 'unprofitable'} with a final PnL of ${pnl:.2f}. The profit factor of {profit_factor:.2f} indicates {profitability} profitability.")

    summary_sheet.write('A10', '3. Consistency Check', bold)
    consistency = "Poor"
    if sharpe_ratio > 1.0:
        consistency = "Excellent"
    elif sharpe_ratio > 0.5:
        consistency = "Good"
    elif sharpe_ratio > 0:
        consistency = "Acceptable"
    summary_sheet.write('A11', f"Consistency appears {consistency}. The Sharpe Ratio of {sharpe_ratio:.2f} indicates the risk-adjusted return. The win rate was {win_rate:.2f}%. ")

    summary_sheet.write('A13', '4. Balanced Verdict', bold)
    verdict = ""
    if pnl > 0 and sharpe_ratio > 0.8 and drawdown_pct < 15:
        verdict = "Excellent. The strategy shows strong, consistent, risk-adjusted returns with low drawdown. It appears robust."
    elif pnl > 0 and sharpe_ratio > 0.5 and drawdown_pct < 25:
        verdict = "Good. The strategy is profitable and reasonably consistent. The risk level is moderate. Consider deploying, but monitor performance closely."
    elif pnl > 0:
        verdict = "Acceptable, with caveats. The strategy is profitable, but may have low consistency (Sharpe Ratio < 0.5) or high risk (Drawdown > 25%). Proceed with caution."
    else:
        verdict = "Not Recommended. The strategy was unprofitable in its current configuration for the tested period."
    summary_sheet.write('A14', verdict)

    summary_sheet.write('A16', '5. Advanced Metrics', bold) 
    
    sortino = performance.get('sortino_ratio', 0)
    calmar = performance.get('calmar_ratio', 0)
    holding_time = performance.get('avg_holding_time', 'N/A')
    max_wins = performance.get('max_consecutive_wins', 0)
    max_losses = performance.get('max_consecutive_losses', 0)

    summary_sheet.write('A17', f"Sortino Ratio: {sortino:.2f} (Measures return against downside risk)")
    summary_sheet.write('A18', f"Calmar Ratio: {calmar:.2f} (Measures return against max drawdown)")
    summary_sheet.write('A19', f"Average Trade Holding Time: {holding_time}")
    summary_sheet.write('A20', f"Max Consecutive Wins: {max_wins}")
    summary_sheet.write('A21', f"Max Consecutive Losses: {max_losses}")

def _generate_portfolio_excel_report(performance, trades_df, symbol_performance_data, filename="ICT_portfolio_results.xlsx", initial_capital=100, days_tested=None, trailing_stop_status=None):
    output_dir = config.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    full_filename = os.path.join(output_dir, filename)
    print(f"Generating Excel report to {full_filename}...")
    try:
        if os.path.exists(full_filename):
            os.remove(full_filename)
        with pd.ExcelWriter(full_filename, engine='xlsxwriter') as writer:
            performance_summary = {k: v for k, v in performance.items() if k != 'equity_curve'}
            performance_summary['Initial Capital'] = initial_capital
            
            report_generated_time = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S")
            performance_summary['Days Tested'] = days_tested
            performance_summary['Trailing Stoploss'] = "On" if trailing_stop_status else "Off"
            performance_summary['Report Generated On'] = report_generated_time
            performance_summary['Biggest Profit'] = performance.get('biggest_profit', 0)
            performance_summary['Biggest Loss'] = performance.get('biggest_loss', 0)
            performance_summary['Trades Invalidated (Max SL)'] = performance.get('max_sl_invalidations', 0)

            summary_df = pd.DataFrame([performance_summary])
            summary_df.to_excel(writer, sheet_name='Portfolio_Report', index=False)
            
            if not trades_df.empty:
                excel_trades_df = trades_df.copy()
                if 'open_time' in excel_trades_df.columns:
                    excel_trades_df['open_time'] = excel_trades_df['open_time'].dt.tz_localize(None)
                if 'close_time' in excel_trades_df.columns:
                    excel_trades_df['close_time'] = excel_trades_df['close_time'].dt.tz_localize(None)
                excel_trades_df.to_excel(writer, sheet_name='Trade_Log', index=False)

            _write_backtest_summary_sheet(writer, performance, initial_capital)

            if symbol_performance_data:
                for spd in symbol_performance_data:
                    spd['Initial Capital'] = initial_capital
                excel_symbol_perf_df = pd.DataFrame(symbol_performance_data).copy()
                if 'avg_holding_time' in excel_symbol_perf_df.columns:
                    excel_symbol_perf_df['avg_holding_time'] = pd.to_timedelta(excel_symbol_perf_df['avg_holding_time']).astype(str)
                excel_symbol_perf_df.to_excel(writer, sheet_name='Symbol_Performance', index=False)

            workbook = writer.book
            worksheet = writer.sheets['Portfolio_Report']
            
            equity_curve = performance.get('equity_curve', [])
            if equity_curve and not trades_df.empty:
                equity_df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
                excel_equity_df = equity_df.copy()
                if 'Time' in excel_equity_df.columns:
                    excel_equity_df['Time'] = pd.to_datetime(excel_equity_df['Time']).dt.tz_localize(None)
                excel_equity_df.to_excel(writer, sheet_name='EquityCurveData', index=False)
                
                chart = workbook.add_chart({'type': 'line'})
                chart.add_series({
                    'name':       'Equity Curve',
                    'categories': f'=EquityCurveData!$A$2:$A${len(equity_curve)+1}',
                    'values':     f'=EquityCurveData!$B$2:$B${len(equity_curve)+1}',
                })
                chart.set_title({'name': 'Equity Curve'})
                chart.set_x_axis({'name': 'Time', 'date_axis': True})
                chart.set_y_axis({'name': 'Account Equity ($)'})
                chart.set_legend({'position': 'none'})
                worksheet.insert_chart('A5', chart, {'x_scale': 2, 'y_scale': 1.5})

        print(f"Successfully saved Excel report: {filename}")
    except Exception as e:
        print(f"Failed to generate Excel report: {e}")

def run_portfolio_backtest(start_date, end_date, trailing_stop_enabled, invalidation_exit_enabled):
    print(f"--- Starting portfolio backtest ---")
    all_portfolio_trades = []
    all_symbol_performance = []
    total_max_sl_invalidations = 0
    
    symbols_to_backtest = config.SYMBOLS
    if not symbols_to_backtest:
        print("No symbols defined in config.SYMBOLS for portfolio backtest. Aborting.")
        return None

    for symbol in symbols_to_backtest:
        print(f"Running backtest for individual symbol: {symbol}")
        
        _, symbol_trades_df, symbol_equity_df, _, max_sl_invalidations_for_symbol = run_backtest(
            symbol_to_test=symbol,
            start_date=start_date,
            end_date=end_date,
            trailing_stop_enabled=trailing_stop_enabled,
            invalidation_exit_enabled=invalidation_exit_enabled,
            generate_plot=False
        )
        total_max_sl_invalidations += max_sl_invalidations_for_symbol
        
        if symbol_trades_df is not None and not symbol_trades_df.empty:
            all_portfolio_trades.append(symbol_trades_df)
            symbol_performance = calculate_performance_metrics(symbol_trades_df, symbol_equity_df.reset_index().values.tolist(), [], INITIAL_EQUITY, max_sl_invalidations_for_symbol)
            symbol_performance['symbol'] = symbol
            custom_risk_params = strategy.get_symbol_specific_setting(config.CUSTOM_RISK_PARAMETERS, symbol)
            tp_rule = custom_risk_params.get('tp_rule', config.TP_RULE) if custom_risk_params else config.TP_RULE
            symbol_performance['TP Rule'] = tp_rule
            all_symbol_performance.append(symbol_performance)
        else:
            print(f"No trades generated for {symbol} in portfolio backtest.")

    if not all_portfolio_trades:
        print("No trades generated across all symbols for portfolio backtest.")
        return None

    portfolio_trades_df = pd.concat(all_portfolio_trades, ignore_index=True)
    portfolio_trades_df.sort_values(by='open_time', inplace=True)
    
    print(f"Total trades in portfolio backtest: {len(portfolio_trades_df)}")

    portfolio_equity_curve = [(start_date.replace(tzinfo=None), INITIAL_EQUITY)]
    current_equity = INITIAL_EQUITY
    for index, trade in portfolio_trades_df.iterrows():
        current_equity += trade['profit']
        portfolio_equity_curve.append((trade['close_time'], current_equity))

    portfolio_performance = calculate_performance_metrics(portfolio_trades_df, portfolio_equity_curve, [], INITIAL_EQUITY, total_max_sl_invalidations)
    
    print(f"--- Portfolio Backtest Results ---")
    print(f"Total Trades: {portfolio_performance.get('total_trades', 0)}")
    print(f"Net PnL: ${portfolio_performance.get('pnl', 0):.2f}")
    print(f"Max Drawdown: ${portfolio_performance.get('max_drawdown_dollars', 0):.2f} ({portfolio_performance.get('max_drawdown_pct', 0):.2f}%)")

    _generate_portfolio_excel_report(
        performance=portfolio_performance, 
        trades_df=portfolio_trades_df, 
        symbol_performance_data=all_symbol_performance,
        initial_capital=INITIAL_EQUITY,
        days_tested=(end_date - start_date).days,
        trailing_stop_status=trailing_stop_enabled
    )

    return portfolio_performance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ICT Bias Bot Backtester')
    parser.add_argument('--mode', type=str, default='backtest', choices=['backtest', 'portfolio'], help='The mode to run the backtester in.')
    parser.add_argument('--days', type=int, help='The number of days to backtest.')
    parser.add_argument('--symbol', type=str, help='The symbol to backtest.')
    parser.add_argument('--plot', action='store_true', help='Generate a plot of the backtest results.')
    args = parser.parse_args()

    # The script used to redirect stdout and stderr to a file. This has been removed.
    # Now, you can redirect the output as you wish from the command line.
    # For example, to save the output to a file named 'my_output.txt', you can run:
    # python backtest.py --symbol EURUSD --days 30 --plot > my_output.txt 2>&1

    try:
        if args.days:
            end_date = datetime.now(pytz.UTC)
            start_date = end_date - timedelta(days=args.days)
        else:
            end_date = datetime(2025, 9, 13, tzinfo=pytz.UTC)
            start_date = end_date - timedelta(days=DEFAULT_BACKTEST_DAYS)
        
        if args.mode == 'portfolio':
            run_portfolio_backtest(
                start_date=start_date,
                end_date=end_date,
                trailing_stop_enabled=config.TRAILING_STOP_LOSS['backtest_enabled'],
                invalidation_exit_enabled=config.ENABLE_INVALIDATION_EXIT
            )
        else:
            run_backtest(
                symbol_to_test=args.symbol,
                start_date=start_date,
                end_date=end_date,
                trailing_stop_enabled=config.TRAILING_STOP_LOSS['backtest_enabled'],
                invalidation_exit_enabled=config.ENABLE_INVALIDATION_EXIT,
                generate_plot=args.plot
            )
    except Exception as e:
        print("--- AN ERROR OCCURRED ---")
        import traceback
        traceback.print_exc()
