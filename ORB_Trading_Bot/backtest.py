# -*- coding: utf-8 -*-
"""
Backtesting engine for the ORB Trading Bot.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import argparse
import matplotlib.pyplot as plt
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

import strategy
import config
from api_client import BrokerClient
from main import get_symbol_market_type
INITIAL_EQUITY = config.INITIAL_EQUITY_BACKTEST
DEFAULT_BACKTEST_DAYS = 30

def plot_results(equity_df, symbol):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
    fig.suptitle(f'Backtest Results: {symbol}', fontsize=16)

    ax1.plot(equity_df.index, equity_df['equity'], label='Equity', color='blue')
    ax1.set_ylabel('Value ($)')
    ax1.set_title('Equity Curve')
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
    
    output_dir = r'D:\Microsoft VS Code\Projects\2025\ORB_Outputs'
    output_path = os.path.join(output_dir, f'{symbol}_backtest_chart.png')
    plt.savefig(output_path)
    print(f"Generated backtest chart for {symbol} at {output_path}")

def plot_pnl_histogram(results_df, symbol):
    """Generates and saves a histogram of profit/loss distribution."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['pnl'], bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Profit/Loss Distribution for {symbol}')
    plt.xlabel('Profit/Loss ($)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    output_dir = r'D:\Microsoft VS Code\Projects\2025\ORB_Outputs'
    file_path = os.path.join(output_dir, f'{symbol}_pnl_histogram.png')
    
    plt.savefig(file_path)
    plt.close()
    print(f"Generated P/L histogram for {symbol}: {file_path}")

def plot_trades_on_price_chart(price_data, trades_df, symbol):
    """
    Plots price and simulated trades on a single chart.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(22, 10))

    # Plot price data
    ax.plot(price_data.index, price_data['Close'], label='Price', color='lightgray', linewidth=1)

    # Plot trades
    for _, trade in trades_df.iterrows():
        open_time = pd.to_datetime(trade['open_time'])
        close_time = pd.to_datetime(trade['close_time'])
        entry_price = trade['entry_price']
        sl_price = trade['sl']
        tp_price = trade['tp']
        trade_type = trade['type']

        color = 'green' if trade_type == 'BUY' else 'red'
        marker = '^' if trade_type == 'BUY' else 'v'

        # Plot entry marker
        ax.plot(open_time, entry_price, marker=marker, color=color, markersize=1, label=f'{trade_type} Entry')

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
    output_dir = r'D:\Microsoft VS Code\Projects\2025\ORB_Outputs'
    plot_filename = os.path.join(output_dir, f"{symbol.replace(' ', '_').replace('(', '').replace(')', '')}_trade_analysis.png")
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Backtest trade analysis chart saved as {plot_filename}.")

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

def calculate_performance_metrics(trades_df, equity_curve, initial_equity):
    total_trades = len(trades_df)
    if total_trades == 0:
        return {}

    equity_df = pd.DataFrame(equity_curve, columns=['time', 'equity'])
    equity_df['time'] = pd.to_datetime(equity_df['time'], utc=True)
    equity_df = equity_df.set_index('time')
    
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0

    gross_profit = wins['pnl'].sum()
    gross_loss = abs(losses['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 0
    
    total_pnl = trades_df['pnl'].sum()
    
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
    max_consecutive_wins, max_consecutive_losses = _calculate_consecutive_streaks(trades_df['pnl'])

    sl_hits = len(trades_df[trades_df['reason'] == 'SL Hit'])
    tp_hits = len(trades_df[trades_df['reason'] == 'TP Hit'])

    biggest_profit = wins['pnl'].max() if not wins.empty else 0
    biggest_loss = losses['pnl'].min() if not losses.empty else 0

    return {
        'total_trades': total_trades,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown_pct,
        'max_drawdown_dollars': max_drawdown_dollars,
        'sharpe_ratio': sharpe_ratio,
        'pnl': total_pnl,
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
        'biggest_profit': biggest_profit,
        'biggest_loss': biggest_loss,
    }

def run_backtest(symbol, start_date, end_date, generate_plot=True, orb_timeframe=None, initial_equity=config.INITIAL_EQUITY_BACKTEST, rr_standard=None, rr_reversal=None):
    """Simulates the ORB strategy on historical data."""
    print("GEMINI IS RUNNING THIS BACKTEST")
    print(f"\n--- Starting Backtest for {symbol} from {start_date} to {end_date} ---")
    
    timeframe = orb_timeframe if orb_timeframe is not None else config.ORB_TIMEFRAME
    risk_reward_standard = rr_standard if rr_standard is not None else config.RISK_REWARD_RATIO_STANDARD
    risk_reward_reversal = rr_reversal if rr_reversal is not None else config.RISK_REWARD_RATIO_REVERSAL

    client = BrokerClient()
    all_data = client.get_data(symbol, timeframe, start_date, end_date)
    client.shutdown()
    
    if all_data.empty:
        print("No data available for backtesting period.")
        return None, pd.DataFrame()

    trades = []
    equity = initial_equity
    equity_curve = [(start_date, initial_equity)]
    market_type = get_symbol_market_type(symbol)
    
    unique_days = all_data.index.normalize().unique()

    for day in unique_days:
        df_day = all_data[all_data.index.date == day.date()]
        if df_day.empty:
            continue

        orh, orl = None, None
        if market_type == 'US':
            orh, orl = strategy.get_opening_range(df_day, config.US_MARKET_OPEN_TIME)
        elif market_type == 'EUROPEAN':
            orh, orl = strategy.get_opening_range(df_day, config.EUROPEAN_MARKET_OPEN_TIME)
        elif market_type == '24_HOUR':
            orh, orl = strategy.get_opening_range_24_hour(df_day, day)

        if orh is None:
            continue
            
        trade_taken_today = False
        bullish_break = False
        bearish_break = False
        
        for i in range(1, len(df_day)):
            if trade_taken_today:
                break
            
            current_df = df_day.iloc[:i+1]
            signals, bullish_break, bearish_break = strategy.check_trade_signals(symbol, current_df, orh, orl, bullish_break, bearish_break)

            if signals:
                for signal in signals:
                    risk = abs(signal['entry'] - signal['sl'])
                    pnl = 0
                    reason = None
                    close_time = None

                    day_slice = df_day.iloc[i+1:]
                    if signal['type'] == 'BUY':
                        tp_hit_time = day_slice[day_slice['High'] >= signal['tp']].index.min()
                        sl_hit_time = day_slice[day_slice['Low'] <= signal['sl']].index.min()
                        if pd.notna(tp_hit_time) and (pd.isna(sl_hit_time) or tp_hit_time <= sl_hit_time):
                            pnl = risk * risk_reward_standard if signal['strategy'] == 'Standard' else risk * risk_reward_reversal
                            reason = 'TP Hit'
                            close_time = tp_hit_time
                        elif pd.notna(sl_hit_time):
                            pnl = -risk
                            reason = 'SL Hit'
                            close_time = sl_hit_time
                    elif signal['type'] == 'SELL':
                        tp_hit_time = day_slice[day_slice['Low'] <= signal['tp']].index.min()
                        sl_hit_time = day_slice[day_slice['High'] >= signal['sl']].index.min()
                        if pd.notna(tp_hit_time) and (pd.isna(sl_hit_time) or tp_hit_time <= sl_hit_time):
                            pnl = risk * risk_reward_standard if signal['strategy'] == 'Standard' else risk * risk_reward_reversal
                            reason = 'TP Hit'
                            close_time = tp_hit_time
                        elif pd.notna(sl_hit_time):
                            pnl = -risk
                            reason = 'SL Hit'
                            close_time = sl_hit_time

                    if pnl != 0:
                        trade = {
                            'symbol': symbol,
                            'open_time': current_df.index[-1],
                            'close_time': close_time,
                            'type': signal['type'],
                            'entry_price': signal['entry'],
                            'sl': signal['sl'],
                            'tp': signal['tp'],
                            'pnl': pnl,
                            'reason': reason,
                            'strategy': signal['strategy']
                        }
                        trades.append(trade)
                        equity += pnl
                        equity_curve.append((close_time, equity))
                        trade_taken_today = True
                        break

    if not trades:
        print("No trades were executed during the backtest period.")
        return None, pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    performance = calculate_performance_metrics(trades_df, equity_curve, initial_equity)

    print("\n--- Backtest Performance Summary ---")
    for metric, value in performance.items():
        print(f"{metric.replace('_', ' ').title()}: {value}")
    
    if generate_plot:
        equity_df = pd.DataFrame(equity_curve, columns=['time', 'equity']).set_index('time')
        plot_results(equity_df, symbol)
        plot_pnl_histogram(trades_df, symbol)
        plot_trades_on_price_chart(all_data, trades_df, symbol)
        plot_trades_on_price_chart(all_data, trades_df, symbol)

    return performance, trades_df

def _write_summary_sheet(writer, performance, initial_capital):
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

def generate_orb_excel_report(performance, trades_df, symbol_performance_data, filename="ORB_portfolio_results.xlsx", initial_capital=100, days_tested=None):
    print(f"Generating Excel report to {filename}...")
    try:
        if os.path.exists(filename):
            os.remove(filename)
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            performance_summary = {k: v for k, v in performance.items() if k != 'equity_curve'}
            performance_summary['Initial Capital'] = initial_capital
            
            report_generated_time = datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
            performance_summary['Days Tested'] = days_tested
            performance_summary['Report Generated On'] = report_generated_time
            performance_summary['Biggest Profit'] = performance.get('biggest_profit', 0)
            performance_summary['Biggest Loss'] = performance.get('biggest_loss', 0)

            summary_df = pd.DataFrame([performance_summary])
            summary_df.to_excel(writer, sheet_name='Portfolio_Report', index=False)
            
            if not trades_df.empty:
                excel_trades_df = trades_df.copy()
                if 'open_time' in excel_trades_df.columns:
                    excel_trades_df['open_time'] = pd.to_datetime(excel_trades_df['open_time']).dt.tz_localize(None)
                if 'close_time' in excel_trades_df.columns:
                    excel_trades_df['close_time'] = pd.to_datetime(excel_trades_df['close_time']).dt.tz_localize(None)
                excel_trades_df.to_excel(writer, sheet_name='Trade_Log', index=False)

            _write_summary_sheet(writer, performance, initial_capital)

            if symbol_performance_data:
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

def run_portfolio_backtest(start_date, end_date):
    """Runs a backtest on all symbols defined in config.py and aggregates the results."""
    print("--- Starting Portfolio Backtest ---")
    
    all_trades = []
    all_symbol_performance = []
    
    symbols_to_test = config.SYMBOLS
    if not symbols_to_test:
        print("No symbols found in config.SYMBOLS. Aborting portfolio backtest.")
        return

    for symbol in symbols_to_test:
        print(f"\n--- Backtesting Symbol: {symbol} ---")
        performance, trades_df = run_backtest(symbol, start_date, end_date, generate_plot=False, initial_equity=config.INITIAL_EQUITY_BACKTEST)
        if trades_df is not None and not trades_df.empty:
            all_trades.append(trades_df)
            
            symbol_equity_curve = performance.get('equity_curve', [])
            symbol_performance = calculate_performance_metrics(trades_df, symbol_equity_curve, config.INITIAL_EQUITY_BACKTEST)
            symbol_performance['symbol'] = symbol
            all_symbol_performance.append(symbol_performance)

    if not all_trades:
        print("No trades were executed across all symbols in the portfolio backtest.")
        return

    portfolio_trades_df = pd.concat(all_trades, ignore_index=True)
    portfolio_trades_df = portfolio_trades_df.sort_values(by='open_time').reset_index(drop=True)

    initial_equity = config.INITIAL_EQUITY_BACKTEST
    equity = initial_equity
    equity_curve = [(portfolio_trades_df['open_time'].iloc[0] if not portfolio_trades_df.empty else start_date, initial_equity)]

    for index, trade in portfolio_trades_df.iterrows():
        equity += trade['pnl']
        equity_curve.append((trade['close_time'], equity))

    portfolio_performance = calculate_performance_metrics(portfolio_trades_df, equity_curve, initial_equity)
    portfolio_performance['equity_curve'] = equity_curve

    output_dir = r'D:\Microsoft VS Code\Projects\2025\ORB_Outputs'
    report_path = os.path.join(output_dir, 'ORB_portfolio_results.xlsx')
    generate_orb_excel_report(
        performance=portfolio_performance, 
        trades_df=portfolio_trades_df, 
        symbol_performance_data=all_symbol_performance,
        filename=report_path,
        initial_capital=initial_equity,
        days_tested=(end_date - start_date).days
    )

    equity_df = pd.DataFrame(equity_curve, columns=['time', 'equity'])
    equity_df['time'] = pd.to_datetime(equity_df['time'], utc=True)
    equity_df = equity_df.set_index('time')
    plot_results(equity_df, 'Portfolio')
    plot_pnl_histogram(portfolio_trades_df, 'Portfolio')



if __name__ == "__main__":
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = None
    try:
        output_file_path = r'D:\Microsoft VS Code\Projects\2025\ORB_Outputs\ORB_Backtest_output.txt'
        log_file = open(output_file_path, 'w')
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = sys.stdout

        try:
            parser = argparse.ArgumentParser(description='ORB Trading Bot Backtester')
            parser.add_argument('--symbol', type=str, help='The symbol to backtest. If not provided, and --portfolio is not used, it defaults to US SP 500.')
            parser.add_argument('--days', type=int, default=DEFAULT_BACKTEST_DAYS, help='The number of days to backtest.')
            parser.add_argument('--plot', action='store_true', help='Generate a plot of the backtest results.')
            parser.add_argument('--portfolio', action='store_true', help='Run a portfolio backtest on all symbols in config.py.')
            args = parser.parse_args()

            end_date = datetime.now(pytz.utc)
            start_date = end_date - timedelta(days=args.days)

            if args.portfolio:
                run_portfolio_backtest(start_date, end_date)
            else:
                symbol = args.symbol if args.symbol else 'US SP 500'
                run_backtest(symbol, start_date, end_date, args.plot, initial_equity=config.INITIAL_EQUITY_BACKTEST)
        except Exception as e:
            import traceback
            print("An error occurred:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_file:
            log_file.close()