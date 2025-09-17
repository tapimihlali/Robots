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

# --- WALK-FORWARD OPTIMIZATION CONFIGURATION ---
# Overall period for Walk-Forward Optimization 2024, 1, 1,
WFO_START_DATE = datetime(2021, 1, 1, tzinfo=pytz.UTC)
WFO_END_DATE = datetime(2025, 9, 10, tzinfo=pytz.UTC)

# Size of the In-Sample (optimization) window in days
IS_WINDOW_DAYS = 1825 # e.g., 180 days for 6 months

# Size of the Out-of-Sample (testing) window in days
OOS_WINDOW_DAYS = 1095 # e.g., 90 days for 3 months

# How many days to advance the window for the next iteration
STEP_DAYS = 365 # e.g., 30 days to slide the window forward by a month

# --- OPTIMIZATION CONFIGURATION ---

OPTIMIZATION_PARAMS = {
    'XAUUSD': {
        'MAX_SL_PIPS': [50, 75, 100],
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
    
    # 'AMZN': {
    #     'MAX_SL_PIPS': [50, 75, 100],
    #     'SL_BUFFER_PIPS': [10, 20],
    #     'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    # },
    # 'AAPL': {
    #     'MAX_SL_PIPS': [50, 75, 100],
    #     'SL_BUFFER_PIPS': [10, 20],
    #     'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    # },
    # 'TSLA': {
    #     'MAX_SL_PIPS': [50, 75, 100],
    #     'SL_BUFFER_PIPS': [10, 20],
    #     'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    # },
    # 'MSFT': {
    #     'MAX_SL_PIPS': [50, 75, 100],
    #     'SL_BUFFER_PIPS': [10, 20],
    #     'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    # },
    
    # 'NGAS': {
    #     'MAX_SL_PIPS': [50, 75, 100],
    #     'SL_BUFFER_PIPS': [10, 20],
    #     'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    # },
    # 'US Oil': {
    #     'MAX_SL_PIPS': [50, 75, 100],
    #     'SL_BUFFER_PIPS': [10, 20],
    #     'TP_RULE': ['1:2', '1:3', 'PDH/PDL']
    # },

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
    # 1. Risk (Max Drawdown): Higher score for lower drawdown.
    # We use an exponential scale to heavily penalize high drawdowns.
    df['drawdown_score'] = 100 * (1 - (abs(df['Max Drawdown (%)']) / 100) ** 0.5)

    # 2. Risk-Adjusted Return (Sharpe Ratio): Higher score for higher Sharpe.
    # Normalize Sharpe, capping at a reasonable level (e.g., 3.0)
    df['sharpe_score'] = (df['Sharpe Ratio'].clip(0, 3) / 3) * 100

    # 3. Profitability (Net Profit & Profit Factor)
    # We reward profitability but at a lower weight than risk metrics.
    df['profit_score'] = (df['Net Profit'] > 0).astype(int) * 50
    df['profit_factor_score'] = (df['Profit Factor'].clip(0, 5) / 5) * 50
    df['profitability_score'] = df['profit_score'] + df['profit_factor_score']

    # 4. Consistency (Win Rate)
    df['win_rate_score'] = (df['Win Rate (%)'] / 100) * 100

    # --- OVERALL SCORE & RANKING ---
    # Weights can be adjusted to prioritize different aspects.
    # Current weights: Drawdown (40%), Sharpe (30%), Profitability (20%), Win Rate (10%)
    weights = {'drawdown': 0.40, 'sharpe': 0.30, 'profitability': 0.20, 'win_rate': 0.10}
    
    df['Overall Score'] = (
        df['drawdown_score'] * weights['drawdown'] + 
        df['sharpe_score'] * weights['sharpe'] + 
        df['profitability_score'] * weights['profitability'] + 
        df['win_rate_score'] * weights['win_rate']
    )

    # Rank within each symbol group
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

    # Drop intermediate score columns
    df.drop(columns=['drawdown_score', 'sharpe_score', 'profit_score', 'profit_factor_score', 'profitability_score', 'win_rate_score'], inplace=True)

    return df

def _run_single_backtest(symbol, param_dict, start_date, end_date):
    """
    Helper function to run a single backtest for parallel execution.
    """
    overrides = {
        'MAX_SL_PIPS': {'default': param_dict['MAX_SL_PIPS']},
        'SL_BUFFER_PIPS': {'default': param_dict['SL_BUFFER_PIPS']},
        'TP_RULE': param_dict['TP_RULE']
    }
    
    # Get trailing stop and invalidation exit settings from config
    trailing_stop_enabled = config.TRAILING_STOP_LOSS['backtest_enabled']
    invalidation_exit_enabled = config.ENABLE_INVALIDATION_EXIT

    summary_result = run_backtest(symbol_to_test=symbol, param_overrides=overrides, generate_plot=False, start_date=start_date, end_date=end_date, trailing_stop_enabled=trailing_stop_enabled, invalidation_exit_enabled=invalidation_exit_enabled, mode='walk_forward')
    if summary_result:
        result_data = summary_result[0]
        run_summary = {**param_dict, **result_data, 'Symbol': symbol,
                       'Trailing Stop Enabled': trailing_stop_enabled,
                       'Invalidation Exit Enabled': invalidation_exit_enabled}
        return run_summary
    return None

def run_walk_forward_optimization():
    """
    Runs a walk-forward optimization, saves results to CSV and Excel, and prints summaries.
    """
    original_stdout = sys.stdout
    output_dir = config.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'optimizer_output.txt')
    log_file = open(output_file_path, 'w')
    
    # Redirect stdout to both console and file
    sys.stdout = Tee(original_stdout, log_file)

    try:
        print("--- STARTING WALK-FORWARD OPTIMIZATION ---")
        
        temp_results_file = os.path.join(output_dir, "ICT_walk_forward_results_temp.csv")
        try:
            all_wfo_results_df = pd.read_csv(temp_results_file)
            all_wfo_results = all_wfo_results_df.to_dict('records')
            # Resume from the last completed step
            last_oos_end_str = all_wfo_results[-1]['WFO_OOS_End']
            # Correctly parse the date string which might be in 'YYYY-MM-DD HH:MM:SS' format
            last_oos_end = pd.to_datetime(last_oos_end_str).tz_localize('UTC')
            current_wfo_start = last_oos_end - timedelta(days=OOS_WINDOW_DAYS - STEP_DAYS)
            print(f"Loaded {len(all_wfo_results)} existing walk-forward results. Resuming from {current_wfo_start.strftime('%Y-%m-%d')}.")
        except (FileNotFoundError, KeyError, IndexError):
            all_wfo_results = []
            current_wfo_start = WFO_START_DATE
            print("No existing temporary results found. Starting fresh.")

        while current_wfo_start + timedelta(days=IS_WINDOW_DAYS + OOS_WINDOW_DAYS) <= WFO_END_DATE:
            is_start = current_wfo_start
            is_end = is_start + timedelta(days=IS_WINDOW_DAYS)
            oos_start = is_end
            oos_end = oos_start + timedelta(days=OOS_WINDOW_DAYS)

            if oos_end > WFO_END_DATE:
                print(f"Skipping WFO step: OOS window ({oos_start.strftime('%Y-%m-%d')} to {oos_end.strftime('%Y-%m-%d')}) exceeds WFO_END_DATE.")
                break

            print(f"\n--- WFO Step: IS [{is_start.strftime('%Y-%m-%d')} - {is_end.strftime('%Y-%m-%d')}] | OOS [{oos_start.strftime('%Y-%m-%d')} - {oos_end.strftime('%Y-%m-%d')}] ---")
        
            # --- IN-SAMPLE OPTIMIZATION ---
            is_results_df = pd.DataFrame()
            for symbol, params in OPTIMIZATION_PARAMS.items():
                param_combinations = list(product(*params.values()))
                tasks = [dict(zip(params.keys(), combo)) for combo in param_combinations]
                
                print(f"Optimizing {symbol} for IS period...")
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(_run_single_backtest, symbol, param_dict, is_start, is_end) for param_dict in tasks]
                    symbol_is_results = [future.result() for future in futures if future.result() is not None]
                
                if symbol_is_results:
                    is_results_df = pd.concat([is_results_df, pd.DataFrame(symbol_is_results)], ignore_index=True)
            
            if is_results_df.empty:
                print(f"No IS results for this WFO step. Advancing window.")
                current_wfo_start += timedelta(days=STEP_DAYS)
                continue

            # Analyze IS results to find best parameters
            is_results_df = analyze_and_suggest(is_results_df)
            best_params_per_symbol = is_results_df.loc[is_results_df.groupby('Symbol')['Overall Score'].idxmax()]

            # --- OUT-OF-SAMPLE TESTING ---
            for index, row in best_params_per_symbol.iterrows():
                symbol = row['Symbol']
                best_param_dict = {k: row[k] for k in OPTIMIZATION_PARAMS[symbol].keys()}
                
                print(f"Testing {symbol} OOS with best IS params...")
                oos_summary = _run_single_backtest(symbol, best_param_dict, oos_start, oos_end)
                if oos_summary:
                    oos_summary['WFO_IS_Start'] = is_start
                    oos_summary['WFO_IS_End'] = is_end
                    oos_summary['WFO_OOS_Start'] = oos_start
                    oos_summary['WFO_OOS_End'] = oos_end
                    all_wfo_results.append(oos_summary)
                    # --- Incremental Save ---
                    pd.DataFrame(all_wfo_results).to_csv(temp_results_file, index=False)
                    print(f"Incrementally saved {len(all_wfo_results)} results to {temp_results_file}")
            
            current_wfo_start += timedelta(days=STEP_DAYS)
        
        if all_wfo_results:
            final_wfo_df = pd.DataFrame(all_wfo_results)
            
            # --- SAVE RAW RESULTS ---
            final_wfo_df.to_csv(os.path.join(output_dir, "ICT_walk_forward_results.csv"), index=False)
            print(f"\nDetailed WFO results saved to {os.path.join(output_dir, "ICT_walk_forward_results.csv")}")

            # --- AGGREGATE AND ANALYZE OOS RESULTS ---
            overall_oos_profit = final_wfo_df['Net Profit'].sum()
            overall_oos_trades = final_wfo_df['Total Trades'].sum()
            print(f"\n--- WALK-FORWARD OPTIMIZATION COMPLETE ---")
            print(f"Total OOS Net Profit: ${overall_oos_profit:.2f}")
            print(f"Total OOS Trades: {overall_oos_trades}")

            analyzed_wfo_df = analyze_and_suggest(final_wfo_df.copy())
            analyzed_wfo_df = analyzed_wfo_df.sort_values(by=['Symbol', 'Rank'])

            best_params_overall = analyzed_wfo_df.sort_values(by='Overall Score', ascending=False)

            # --- SAVE TO EXCEL ---
            excel_final_wfo_df = final_wfo_df.copy()
            for col in ['WFO_IS_Start', 'WFO_IS_End', 'WFO_OOS_Start', 'WFO_OOS_End']:
                if col in excel_final_wfo_df.columns:
                    excel_final_wfo_df[col] = pd.to_datetime(excel_final_wfo_df[col]).dt.tz_localize(None)

            excel_analyzed_wfo_df = analyzed_wfo_df.copy()
            for col in ['WFO_IS_Start', 'WFO_IS_End', 'WFO_OOS_Start', 'WFO_OOS_End']:
                if col in excel_analyzed_wfo_df.columns:
                    excel_analyzed_wfo_df[col] = pd.to_datetime(excel_analyzed_wfo_df[col]).dt.tz_localize(None)

            excel_best_params_overall = best_params_overall.copy()
            for col in ['WFO_IS_Start', 'WFO_IS_End', 'WFO_OOS_Start', 'WFO_OOS_End']:
                if col in excel_best_params_overall.columns:
                    excel_best_params_overall[col] = pd.to_datetime(excel_best_params_overall[col]).dt.tz_localize(None)

            excel_output_path = os.path.join(output_dir, "ICT_walk_forward_results.xlsx")
            with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
                excel_final_wfo_df.to_excel(writer, sheet_name='Walk-Forward Raw Results', index=False)
                excel_analyzed_wfo_df.to_excel(writer, sheet_name='Optimization Analysis', index=False)
                excel_best_params_overall.to_excel(writer, sheet_name='Best Parameters Overall', index=False)
            
            print("Walk-Forward results, analysis, and best parameters saved to ICT_walk_forward_results.xlsx")
            # --- CLEANUP ---
            if os.path.exists(temp_results_file):
                os.remove(temp_results_file)
                print(f"Removed temporary results file: {temp_results_file}")

        else:
            print("No Walk-Forward Optimization results were generated.")

    finally:
        # Restore original stdout and close the file
        sys.stdout = original_stdout
        log_file.close()

if __name__ == "__main__":
    run_walk_forward_optimization()
