#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.stats import mode

import os
import deriv_mt5_bot_config as config
from deriv_mt5_bot_v2 import TradingBot

# --- OUTPUT DIRECTORY ---
OUTPUT_DIR = "D:\\Microsoft VS Code\\Projects\\2025\\Deriv_EMA_Outputs"

# --- Bot Configuration and Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "MT5_walk_forward_optimizer_output.txt"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_walk_forward_optimization(bot, symbol, total_days, in_sample_days, out_of_sample_days, initial_capital, use_trailing_stop):
    """Performs walk-forward optimization for a given symbol."""
    logger.info(f"--- Starting Walk-Forward Optimization for {symbol} ---")
    logger.info(f"Total Period: {total_days} days | In-Sample: {in_sample_days} days | Out-of-Sample: {out_of_sample_days} days")
    logger.info(f"Initial Capital: ${initial_capital} | Trailing Stop: {'Enabled' if use_trailing_stop else 'Disabled'}")

    # --- Download all data at once ---
    logger.info(f"Downloading {total_days} days of historical data for {symbol}...")
    full_data = bot._prepare_data_for_backtest(symbol, days=total_days)
    if full_data.empty:
        logger.error(f"Could not download historical data for {symbol}. Aborting.")
        return
    logger.info("Historical data download complete.")

    # --- Parameter Ranges ---
    sl_multiplier_range = np.arange(1.0, 3.1, 0.2)
    tp_risk_reward_range = np.arange(1.0, 4.1, 0.2)

    all_out_of_sample_results = []
    
    n_runs = int((total_days - in_sample_days) / out_of_sample_days)
    if n_runs <= 0:
        logger.error("Total days must be greater than in-sample days.")
        return

    for i in range(n_runs):
        end_date = full_data.index[-1] - timedelta(days=i * out_of_sample_days)
        out_of_sample_end = end_date
        in_sample_end = out_of_sample_end - timedelta(days=out_of_sample_days)
        in_sample_start = in_sample_end - timedelta(days=in_sample_days)

        logger.info(f"\n--- WFO Run {i+1}/{n_runs} for {symbol} ---")
        logger.info(f"In-Sample Period: {in_sample_start.date()} to {in_sample_end.date()}")
        logger.info(f"Out-of-Sample Period: {in_sample_end.date()} to {out_of_sample_end.date()}")

        in_sample_data = full_data.loc[in_sample_start:in_sample_end]
        out_of_sample_data = full_data.loc[in_sample_end:out_of_sample_end]

        if in_sample_data.empty or out_of_sample_data.empty:
            logger.warning(f"Not enough data for this window (Run {i+1}). Skipping.")
            continue

        # 1. In-Sample Optimization
        best_params = {'sl': None, 'tp': None, 'sharpe': -np.inf}
        for sl in sl_multiplier_range:
            for tp in tp_risk_reward_range:
                trades = bot._run_vectorized_backtest(in_sample_data.copy(), symbol, sl, tp, use_trailing_stop, initial_capital)
                if not trades.empty:
                    metrics = bot.calculate_performance_metrics(trades, initial_capital)
                    if metrics['sharpe_ratio'] > best_params['sharpe']:
                        best_params = {'sl': sl, 'tp': tp, 'sharpe': metrics['sharpe_ratio']}
        
        if best_params['sl'] is None:
            logger.warning(f"No profitable parameters found in-sample for run {i+1}. Skipping.")
            continue
        
        logger.info(f"Best In-Sample Params: SL={best_params['sl']:.1f}, TP={best_params['tp']:.1f}, Sharpe={best_params['sharpe']:.2f}")

        # 2. Out-of-Sample Testing
        oos_trades = bot._run_vectorized_backtest(out_of_sample_data.copy(), symbol, best_params['sl'], best_params['tp'], use_trailing_stop, initial_capital)
        oos_metrics = bot.calculate_performance_metrics(oos_trades, initial_capital)
        
        logger.info(f"Out-of-Sample PnL: ${oos_metrics.get('pnl', 0):.2f}, Sharpe: {oos_metrics.get('sharpe_ratio', 0):.2f}")
        
        result = {
            'Run': i + 1,
            'Symbol': symbol,
            'In-Sample Start': in_sample_start.date(),
            'In-Sample End': in_sample_end.date(),
            'Out-of-Sample End': out_of_sample_end.date(),
            'Best SL': best_params['sl'],
            'Best TP': best_params['tp'],
            'OOS PnL ($)': oos_metrics.get('pnl', 0),
            'OOS Sharpe Ratio': oos_metrics.get('sharpe_ratio', 0),
            'OOS Max Drawdown (%)': oos_metrics.get('max_drawdown_pct', 0),
            'OOS Win Rate': oos_metrics.get('win_rate', 0),
            'OOS Total Trades': oos_metrics.get('total_trades', 0)
        }
        all_out_of_sample_results.append(result)

    # 3. Reporting
    if not all_out_of_sample_results:
        logger.warning("Walk-forward optimization finished with no results.")
        return

    results_df = pd.DataFrame(all_out_of_sample_results)
    
    # --- Calculate Score for Analysis ---
    # Normalize metrics (0-1 scale, higher is better)
    pnl_norm = (results_df['OOS PnL ($)'] - results_df['OOS PnL ($)'].min()) / (results_df['OOS PnL ($)'].max() - results_df['OOS PnL ($)'].min())
    sharpe_norm = (results_df['OOS Sharpe Ratio'] - results_df['OOS Sharpe Ratio'].min()) / (results_df['OOS Sharpe Ratio'].max() - results_df['OOS Sharpe Ratio'].min())
    win_rate_norm = results_df['OOS Win Rate'] / 100
    # Invert drawdown because lower is better
    drawdown_norm = 1 - ((results_df['OOS Max Drawdown (%)'] - results_df['OOS Max Drawdown (%)'].min()) / (results_df['OOS Max Drawdown (%)'].max() - results_df['OOS Max Drawdown (%)'].min()))

    # Calculate weighted score
    results_df['Score'] = (pnl_norm * 0.3) + (sharpe_norm * 0.3) + (win_rate_norm * 0.2) + (drawdown_norm * 0.2)
    results_df['Score'] = results_df['Score'].fillna(0)


    report_path = os.path.join(OUTPUT_DIR, f"MT5_walk_forward_results_{symbol.replace('/', '_')}.xlsx")

    with pd.ExcelWriter(report_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # --- Parameters Sheet ---
        params_sheet = workbook.add_worksheet('Parameters')
        params_sheet.set_column('A:A', 30)
        params_sheet.set_column('B:B', 20)
        header_format = workbook.add_format({'bold': True, 'bg_color': '#DDEBF7', 'border': 1})
        
        params_sheet.write('A1', 'Parameter', header_format)
        params_sheet.write('B1', 'Value', header_format)
        param_data = {
            "Report Generated On": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": symbol,
            "Total Test Duration (Days)": total_days,
            "In-Sample Duration (Days)": in_sample_days,
            "Out-of-Sample Duration (Days)": out_of_sample_days,
            "Initial Capital": f"${initial_capital:,.2f}",
            "Trailing Stoploss": "On" if use_trailing_stop else "Off"
        }
        row = 1
        for key, value in param_data.items():
            params_sheet.write(row, 0, key)
            params_sheet.write(row, 1, value)
            row += 1

        # --- WFO Runs Sheet ---
        results_df.to_excel(writer, sheet_name='WFO_Runs', index=False)

        # --- Analysis and Recommendations Sheet ---
        analysis_sheet = workbook.add_worksheet('Analysis_and_Recommendations')
        analysis_sheet.set_column('A:M', 15)
        
        # Write the dataframe to the sheet
        results_df.sort_values(by='Score', ascending=False).to_excel(writer, sheet_name='Analysis_and_Recommendations', index=False, startrow=6)

        title_format = workbook.add_format({'bold': True, 'font_size': 14})
        subtitle_format = workbook.add_format({'bold': True, 'font_size': 11})
        text_wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})

        analysis_sheet.write('A1', 'Optimization Analysis and Recommendations', title_format)
        analysis_sheet.write('A3', 'Scoring Methodology:', subtitle_format)
        analysis_sheet.write('A4', 'The Score (0-1) provides a balanced measure of run quality. It considers: Sharpe Ratio (30%), Final PnL (30%), Win Rate (20%), and Max Drawdown (20%, inverted). A higher score indicates a better risk-adjusted performance.', text_wrap_format)
        analysis_sheet.merge_range('A4:F5', analysis_sheet.cell_values()[3][0], text_wrap_format)

        # --- Best Parameters Overall Sheet ---
        best_params_sheet = workbook.add_worksheet('Best_Parameters_Overall')
        best_params_sheet.set_column('A:B', 25)
        best_params_sheet.write('A1', 'Most Robust Parameters', title_format)
        best_params_sheet.write('A3', 'These are the most frequently occurring "best" parameters from all the in-sample optimization runs. They can be considered the most robust and adaptive parameters across different market periods.', text_wrap_format)
        best_params_sheet.merge_range('A3:D4', best_params_sheet.cell_values[2][0], text_wrap_format)

        robust_sl = mode(results_df['Best SL']).mode[0]
        robust_tp = mode(results_df['Best TP']).mode[0]
        
        best_params_sheet.write('A6', 'Parameter', header_format)
        best_params_sheet.write('B6', 'Value', header_format)
        best_params_sheet.write('A7', 'Robust SL Multiplier')
        best_params_sheet.write('B7', f"{robust_sl:.2f}")
        best_params_sheet.write('A8', 'Robust TP Risk/Reward')
        best_params_sheet.write('B8', f"{robust_tp:.2f}")

        # --- WFO Summary Sheet ---
        summary_df = results_df.agg({
            'OOS PnL ($)': ['sum', 'mean'],
            'OOS Sharpe Ratio': ['mean'],
            'OOS Max Drawdown (%)': ['mean', 'max'],
            'OOS Win Rate': ['mean']
        })
        summary_df.to_excel(writer, sheet_name='WFO_Summary')

    logger.info(f"\n--- WALK-FORWARD OPTIMIZATION COMPLETE for {symbol} ---")
    logger.info(f"Report saved to {report_path}")
    print(summary_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Walk-Forward Optimizer for the Deriv MT5 Trading Bot")
    parser.add_argument('--symbol', type=str, required=True, help='The symbol to optimize.')
    parser.add_argument('--total_days', type=int, default=365, help='Total period for the walk-forward analysis.')
    parser.add_argument('--in_sample_days', type=int, default=90, help='The length of the in-sample (training) period.')
    parser.add_argument('--out_of_sample_days', type=int, default=30, help='The length of the out-of-sample (testing) period.')
    parser.add_argument('--initial_capital', type=int, default=500, help='The initial capital for backtesting.')
    parser.add_argument('--use_trailing_stop', action='store_true', help='Enable trailing stop in the backtest.')

    args = parser.parse_args()

    credentials = config.get_credentials()
    bot = None
    try:
        bot = TradingBot(user=credentials['user'], password=credentials['password'], server=credentials['server'], path=credentials['path'])
        if bot.mt5_connected:
            run_walk_forward_optimization(
                bot, 
                args.symbol, 
                args.total_days, 
                args.in_sample_days, 
                args.out_of_sample_days,
                args.initial_capital,
                args.use_trailing_stop
            )
        else:
            logger.critical("Optimizer could not connect to MT5. Aborting.")

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user.")
    except Exception as e:
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)
    finally:
        if bot:
            bot.shutdown()
