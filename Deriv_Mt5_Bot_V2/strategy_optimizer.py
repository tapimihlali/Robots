#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

import deriv_mt5_bot_config as config
from deriv_mt5_bot_v2 import TradingBot

# --- OUTPUT DIRECTORY ---
OUTPUT_DIR = "D:\\Microsoft VS Code\\Projects\\2025\\Deriv_EMA_Outputs"

# --- Bot Configuration and Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "MT5_strategy_optimizer_output.txt"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



def run_optimization(bot, symbols_to_optimize, days_to_optimize, initial_capital=100):
    """Runs the optimization process for a given list of symbols and days, saving progress incrementally."""

    # --- PARAMETER RANGES TO TEST ---
    sl_multiplier_range = np.arange(1.0, 3, 0.1)
    tp_risk_reward_range = np.arange(1.0, 3.2, 0.1)

    logger.info(f"Starting optimization for {len(symbols_to_optimize)} symbols over {days_to_optimize} days.")
    logger.info(f"SL Multiplier Range: {sl_multiplier_range}")
    logger.info(f"TP R/R Range: {tp_risk_reward_range}")

    all_results = []
    raw_output_path = os.path.join(OUTPUT_DIR, "MT5_optimization_raw_results.csv")
    final_report_path = os.path.join(OUTPUT_DIR, "MT5_optimization_results.xlsx")

    logger.info(f"Raw optimization progress will be saved continuously to {raw_output_path}")

    # Load existing raw results if the file exists, to resume optimization.
    if os.path.exists(raw_output_path):
        try:
            existing_df = pd.read_csv(raw_output_path)
            all_results = existing_df.to_dict('records')
            logger.info(f"Loaded {len(all_results)} existing raw results from {raw_output_path} to resume.")
        except Exception as e:
            logger.error(f"Could not load existing raw results from {raw_output_path}: {e}. Starting fresh.")
            all_results = []

    total_runs = len(symbols_to_optimize) * len(sl_multiplier_range) * len(tp_risk_reward_range)
    run_count = 0

    for symbol in symbols_to_optimize:
        for sl in sl_multiplier_range:
            for tp in tp_risk_reward_range:
                run_count += 1

                # Check if this combination has already been run
                is_completed = False
                for res in all_results:
                    # Check for exact match of symbol, sl, and tp
                    if (
                        res.get('Symbol') == symbol and
                        res.get('SL Multiplier') == sl and
                        res.get('TP RR') == tp
                    ):
                        is_completed = True
                        break
                if is_completed:
                    logger.info(f"--- [Run {run_count}/{total_runs}] SKIPPING {symbol} | SL: {sl:.1f}, TP: {tp:.1f} (already completed) ---")
                    continue

                logger.info(f"--- [Run {run_count}/{total_runs}] Optimizing {symbol} | SL: {sl:.1f}, TP: {tp:.1f} ---")
                
                performance = bot.run_backtest(
                    symbol=symbol, 
                    days=days_to_optimize, 
                    sl_multiplier=sl, 
                    tp_risk_reward=tp,
                    generate_report=False, # No individual reports during optimization
                    initial_capital=initial_capital
                )

                if performance is None:
                    logger.warning(f"Backtest failed for {symbol} with SL={sl}, TP={tp}. Skipping.")
                    continue

                result = {
                    'Symbol': symbol,
                    'SL Multiplier': sl,
                    'TP RR': tp,
                    'Final PnL ($)': performance.get('pnl', 0),
                    'Sharpe Ratio': performance.get('sharpe_ratio', 0),
                    'Max Drawdown ($)': performance.get('max_drawdown_dollars', 0),
                    'Max Drawdown (%)': performance.get('max_drawdown_pct', 0),
                    'Total Trades': performance.get('total_trades', 0),
                    'Win Rate': performance.get('win_rate', 0),
                    'Initial Capital': initial_capital
                }
                all_results.append(result)

                # Save raw progress to CSV after each completed backtest
                try:
                    pd.DataFrame(all_results).to_csv(raw_output_path, index=False)
                except Exception as e:
                    logger.error(f"Error saving raw progress to {raw_output_path}: {e}")

    if not all_results:
        logger.warning("Optimization finished with no results.")
        return

    # Now, generate the final Excel report from all_results
    results_df = pd.DataFrame(all_results)

    best_params_list = []
    for symbol in symbols_to_optimize:
        symbol_df = results_df[results_df['Symbol'] == symbol]
        if not symbol_df.empty:
            best_row = symbol_df.sort_values(by=['Sharpe Ratio', 'Final PnL ($)'], ascending=False).iloc[0]
            best_params_list.append(best_row)

    logger.info("\n--- OPTIMIZATION COMPLETE ---")
    if best_params_list:
        best_params_df = pd.DataFrame(best_params_list)
        logger.info("Best parameters found for each symbol (sorted by Sharpe Ratio & PnL):")
        print(best_params_df.to_string())

        # --- RECOMMENDATIONS ---
        costly_symbols = best_params_df[
            (best_params_df['Final PnL ($)'] <= 0) |
            (best_params_df['Sharpe Ratio'] < 0.1)
        ]
        
        # --- Save to Excel ---
        try:
            with pd.ExcelWriter(final_report_path, engine='xlsxwriter') as writer:
                best_params_df.to_excel(writer, sheet_name='Best_Parameters_Overall', index=False)

                # --- Enhanced Analysis Sheet ---
                analysis_sheet = writer.book.add_worksheet('Optimization_Analysis')
                bold = writer.book.add_format({'bold': True})
                analysis_sheet.set_column('A:G', 18)

                analysis_sheet.write('A1', 'Optimization Analysis and Recommendations', bold)
                analysis_sheet.write('A2', f"Initial Capital: ${initial_capital:.2f}")
                analysis_sheet.write('A3', f"Days Optimized: {days_to_optimize}")
                analysis_sheet.write('A4', f"Trailing Stoploss: {'On' if config.BACKTEST_TRAILING_STOP else 'Off'}")
                analysis_sheet.write('A5', f"Report Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                analysis_sheet.write('A6', 'This sheet provides a detailed look at the best and worst parameter combinations for each symbol based on a composite score.')
                analysis_sheet.write('A7', 'The score considers: Sharpe Ratio (high), Final PnL (high), Max Drawdown (low), and Win Rate (high).')

                # Normalize the metrics to calculate a score
                results_df['pnl_norm'] = (results_df['Final PnL ($)'] - results_df['Final PnL ($)'].min()) / (results_df['Final PnL ($)'].max() - results_df['Final PnL ($)'].min())
                results_df['sharpe_norm'] = (results_df['Sharpe Ratio'] - results_df['Sharpe Ratio'].min()) / (results_df['Sharpe Ratio'].max() - results_df['Sharpe Ratio'].min())
                results_df['drawdown_norm'] = 1 - ((results_df['Max Drawdown (%)'] - results_df['Max Drawdown (%)'].min()) / (results_df['Max Drawdown (%)'].max() - results_df['Max Drawdown (%)'].min()))
                results_df['winrate_norm'] = (results_df['Win Rate'] - results_df['Win Rate'].min()) / (results_df['Win Rate'].max() - results_df['Win Rate'].min())

                # Calculate composite score (weights can be adjusted)
                results_df['score'] = (0.4 * results_df['sharpe_norm']) + (0.3 * results_df['pnl_norm']) + (0.2 * results_df['drawdown_norm']) + (0.1 * results_df['winrate_norm'])

                row = 9
                for symbol in symbols_to_optimize:
                    analysis_sheet.write(row, 0, f'Analysis for: {symbol}', bold)
                    row += 1

                    symbol_df = results_df[results_df['Symbol'] == symbol].sort_values(by='score', ascending=False)

                    if symbol_df.empty:
                        analysis_sheet.write(row, 0, "No valid results for this symbol.")
                        row += 2
                        continue

                    # Best Parameters
                    analysis_sheet.write(row, 0, 'Top 3 BEST Parameter Sets', bold)
                    best_df = symbol_df.head(3)[
                        ['SL Multiplier', 'TP RR', 'Final PnL ($)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate', 'Initial Capital', 'score']
                    ]
                    best_df.to_excel(writer, sheet_name='Optimization_Analysis', startrow=row, startcol=0, index=False)
                    row += len(best_df) + 2

                    # Worst Parameters
                    analysis_sheet.write(row, 0, 'Top 3 WORST Parameter Sets', bold)
                    worst_df = symbol_df.tail(3).sort_values(by='score', ascending=True)[
                        ['SL Multiplier', 'TP RR', 'Final PnL ($)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate', 'score']
                    ]
                    worst_df.to_excel(writer, sheet_name='Optimization_Analysis', startrow=row, startcol=0, index=False)
                    row += len(worst_df) + 3 # Add extra space

                logger.info(f"Optimization report with summary has been saved to {final_report_path}")

        except Exception as e:
            logger.error(f"Error saving final optimization report to Excel: {e}", exc_info=True)

        # ... (console output for recommendations) ...
        logger.info("\n--- SYMBOL RECOMMENDATIONS ---")
        if not costly_symbols.empty:
            logger.warning("The following symbols performed poorly and might be too costly to trade.")
            logger.warning("Consider removing them from your symbol list or re-evaluating the strategy for them:")
            for index, row in costly_symbols.iterrows():
                logger.warning(f"  - {row['Symbol']}: PnL=${row['Final PnL ($)']:.2f}, Sharpe={row['Sharpe Ratio']:.2f}")
        else:
            logger.info("All optimized symbols appear to be profitable based on the defined criteria.")

    else:
        logger.info("No best parameters could be determined.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Strategy Optimizer for the Deriv MT5 Trading Bot")
    parser.add_argument('--symbols', type=str, nargs='*', 
                        help='Optional: A list of symbols to optimize, separated by spaces. Overrides the config file.')
    # You can change the default number of days for optimization here
    parser.add_argument('--days', type=int, default=1825, # <--- CHANGE THIS VALUE
                        help="The number of days to optimize over. Defaults to the value set in the script.")
    args = parser.parse_args()

    symbols = args.symbols if args.symbols else config.SYMBOLS_TO_OPTIMIZE

    credentials = config.get_credentials()
    MT5_USER = credentials['user']
    MT5_PASS = credentials['password']
    MT5_SERVER = credentials['server']
    MT5_PATH = credentials['path']

    bot = None
    try:
        bot = TradingBot(user=MT5_USER, password=MT5_PASS, server=MT5_SERVER, path=MT5_PATH)
        if bot.mt5_connected:
            run_optimization(bot, symbols, args.days)
        else:
            logger.critical("Optimizer could not connect to MT5. Aborting.")

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user.")
    except Exception as e:
        logger.critical(f"An unhandled exception occurred during optimization: {e}", exc_info=True)
    finally:
        if bot:
            bot.shutdown()
