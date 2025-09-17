import pandas as pd
import numpy as np
from itertools import product
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import pytz
import sys
import os

# Import the refactored backtest runner
from backtest import run_backtest
import config

WFO_START_DATE = datetime(2024, 1, 1, tzinfo=pytz.UTC)
WFO_END_DATE = datetime(2025, 12, 31, tzinfo=pytz.UTC)

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

# --- OPTIMIZATION CONFIGURATION ---

OPTIMIZATION_PARAMS = {
    'XAUUSD': {
        'MAX_SL_PIPS': [50 , 75, 100],
        'SL_BUFFER_PIPS': [10, 20],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },
    'BTCUSD': {
        'MAX_SL_PIPS': [50, 75, 100],
        'SL_BUFFER_PIPS': [10, 20],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },
    'XAGUSD': {
        'MAX_SL_PIPS': [50, 75, 100],
        'SL_BUFFER_PIPS': [10, 20],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },
    'XPTUSD': {
        'MAX_SL_PIPS': [50, 75, 100],
        'SL_BUFFER_PIPS': [10, 20],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },
    
    'US SP 500': {
        'MAX_SL_PIPS': [50, 75, 100],
        'SL_BUFFER_PIPS': [10, 20],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },
    'US Tech 100': {
        'MAX_SL_PIPS': [50, 75, 100],
        'SL_BUFFER_PIPS': [10, 20],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },
    'Wall Street 30': {
        'MAX_SL_PIPS': [50, 75, 100],
        'SL_BUFFER_PIPS': [10, 20],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },
    'Germany 40': {
        'MAX_SL_PIPS': [50, 75, 100],
        'SL_BUFFER_PIPS': [10, 20],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },
    'UK 100': {
        'MAX_SL_PIPS': [50, 75, 100],
        'SL_BUFFER_PIPS': [10, 20],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },

    'USDJPY': {
        'MAX_SL_PIPS': [10, 20, 30],
        'SL_BUFFER_PIPS': [5, 10],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },
    'NZDUSD': {
        'MAX_SL_PIPS': [10, 20, 30],
        'SL_BUFFER_PIPS': [5, 10],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },
    'GBPUSD': {
        'MAX_SL_PIPS': [10, 20, 30],
        'SL_BUFFER_PIPS': [5, 10],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },
    'EURUSD': {
        'MAX_SL_PIPS': [10, 20, 30],
        'SL_BUFFER_PIPS': [5, 10],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },
    'AUDUSD': {
        'MAX_SL_PIPS': [10, 20, 30],
        'SL_BUFFER_PIPS': [5, 10],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },
    'USDCAD': {
        'MAX_SL_PIPS': [10, 20, 30],
        'SL_BUFFER_PIPS': [5, 10],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },
    'USDCHF': {
        'MAX_SL_PIPS': [10, 20, 30],
        'SL_BUFFER_PIPS': [5, 10],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },
    'USDMXN': {
        'MAX_SL_PIPS': [10, 20, 30],
        'SL_BUFFER_PIPS': [5, 10],
        'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    },
}

def analyze_and_suggest(df):
    """
    Analyzes optimization results to provide data-driven suggestions and rankings.
    """
    # --- SCORING LOGIC ---
    df['drawdown_score'] = 100 * (1 - (abs(df['Max Drawdown (%)']) / 100) ** 0.5)
    df['sharpe_score'] = (df['Sharpe Ratio'].clip(0, 3) / 3) * 100
    df['profit_score'] = (df['Net Profit'] > 0).astype(int) * 50
    df['profit_factor_score'] = (df['Profit Factor'].clip(0, 5) / 5) * 50
    df['profitability_score'] = df['profit_score'] + df['profit_factor_score']
    df['win_rate_score'] = (df['Win Rate (%)'] / 100) * 100

    # --- OVERALL SCORE & RANKING ---
    weights = {'drawdown': 0.40, 'sharpe': 0.30, 'profitability': 0.20, 'win_rate': 0.10}
    
    df['Overall Score'] = (
        df['drawdown_score'] * weights['drawdown'] + 
        df['sharpe_score'] * weights['sharpe'] + 
        df['profitability_score'] * weights['profitability'] + 
        df['win_rate_score'] * weights['win_rate']
    )

    df['Rank'] = df.groupby('Symbol')['Overall Score'].rank(method='dense', ascending=False).astype(int)

    # --- SUGGESTION ---
    conditions = [
        (df['Rank'] == 1) & (df['Overall Score'] > 75),
        (df['Rank'] <= 3) & (df['Overall Score'] > 60),
        (df['Net Profit'] > 0) & (df['Profit Factor'] > 1.2),
        (df['Net Profit'] > 0)
    ]
    choices = ['Top Pick', 'Promising', 'Viable', 'Review']
    df['Suggestion'] = np.select(conditions, choices, default='Avoid')

    df.drop(columns=['drawdown_score', 'sharpe_score', 'profit_score', 'profit_factor_score', 'profitability_score', 'win_rate_score'], inplace=True)

    return df

def _run_single_backtest(symbol, param_dict, start_date, end_date):
    """
    Helper function to run a single backtest for parallel execution.
    """
    overrides = {
        'MAX_SL_PIPS': {'default': param_dict['MAX_SL_PIPS']},
        'SL_BUFFER_PIPS': {'default': param_dict['SL_BUFFER_PIPS']},
        'TP_RULE': {'default': param_dict['TP_RULE']}
    }
    
    trailing_stop_enabled = config.TRAILING_STOP_LOSS['backtest_enabled']
    invalidation_exit_enabled = config.ENABLE_INVALIDATION_EXIT

    summary_result = run_backtest(symbol_to_test=symbol, param_overrides=overrides, generate_plot=False, start_date=start_date, end_date=end_date, trailing_stop_enabled=trailing_stop_enabled, invalidation_exit_enabled=invalidation_exit_enabled, mode='optimizer')
    if summary_result and summary_result[0]:
        results_list = summary_result[0]
        if results_list:
            result_data = results_list[0]
            run_summary = {**param_dict, **result_data, 'Symbol': symbol,
                           'Trailing Stop Enabled': trailing_stop_enabled,
                           'Invalidation Exit Enabled': invalidation_exit_enabled}
            return run_summary
    return None

def run_optimization():
    """
    Runs a grid search optimization, saves results incrementally, and provides a data-driven summary.
    """
    original_stdout = sys.stdout
    output_dir = config.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'optimizer_output.txt')
    log_file = open(output_file_path, 'w')
    
    sys.stdout = Tee(original_stdout, log_file)

    try:
        print("--- STARTING STANDARD PARAMETER OPTIMIZATION ---")
        
        csv_output_path = os.path.join(output_dir, "ICT_optimization_raw_results.csv")
        try:
            all_results_df = pd.read_csv(csv_output_path)
            print("Loaded existing optimization results.")
        except FileNotFoundError:
            all_results_df = pd.DataFrame()
            print("No existing results found. Starting fresh.")

        total_symbols = len(OPTIMIZATION_PARAMS)
        symbol_count = 0

        for symbol, params in OPTIMIZATION_PARAMS.items():
            symbol_count += 1
            param_combinations = list(product(*params.values()))
            tasks = [dict(zip(params.keys(), combo)) for combo in param_combinations]
            
            print(f"\n--- Optimizing Symbol {symbol_count}/{total_symbols}: {symbol} ({len(tasks)} combinations) ---")

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(_run_single_backtest, symbol, param_dict, WFO_START_DATE, WFO_END_DATE) for param_dict in tasks]
                symbol_results = [future.result() for future in futures if future.result() is not None]

            if not symbol_results:
                print(f"No results generated for {symbol}. Skipping save.")
                continue

            new_results_df = pd.DataFrame(symbol_results)
            if not all_results_df.empty and 'Symbol' in all_results_df.columns:
                all_results_df = all_results_df[all_results_df['Symbol'] != symbol]
            
            all_results_df = pd.concat([all_results_df, new_results_df], ignore_index=True)
            all_results_df.to_csv(csv_output_path, index=False)
            print(f"Saved results for {symbol} to {csv_output_path}")

        if not all_results_df.empty:
            print("\n--- OPTIMIZATION COMPLETE ---")
            final_df = analyze_and_suggest(all_results_df)
            final_df = final_df.sort_values(by=['Symbol', 'Rank'])

            best_params_summary = final_df.loc[final_df.groupby('Symbol')['Overall Score'].idxmax()]

            excel_output_path = os.path.join(output_dir, "ICT_optimization_results.xlsx")
            with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
                best_params_summary.to_excel(writer, sheet_name='Best Parameters Summary', index=False)
                final_df.to_excel(writer, sheet_name='Full Optimization Results', index=False)
            
            print(f"Data-driven optimization summary and full results saved to {excel_output_path}")
            print("\n--- TOP 3 PICKS PER SYMBOL ---")
            for symbol in final_df['Symbol'].unique():
                print(f"\n--- {symbol} ---")
                top_3 = final_df[final_df['Symbol'] == symbol].head(3)
                print(top_3[['Rank', 'Suggestion', 'MAX_SL_PIPS', 'SL_BUFFER_PIPS', 'TP_RULE', 'Net Profit', 'Sharpe Ratio', 'Max Drawdown (%)']].to_string(index=False))

        else:
            print("\n--- OPTIMIZATION COMPLETE ---")
            print("No results were generated during the optimization.")
    finally:
        sys.stdout = original_stdout
        log_file.close()

if __name__ == "__main__":
    run_optimization()