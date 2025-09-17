# -*- coding: utf-8 -*-
"""
Walk-forward optimization framework for the ORB Trading Bot.
"""

import pandas as pd
from itertools import product
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import pytz
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

from backtest import run_backtest
import config

# --- WALK-FORWARD OPTIMIZATION CONFIGURATION ---
WFO_START_DATE = datetime(2024, 1, 1, tzinfo=pytz.UTC)
WFO_END_DATE = datetime.now(pytz.utc)
IS_WINDOW_DAYS = 180
OOS_WINDOW_DAYS = 90
STEP_DAYS = 30

OPTIMIZATION_PARAMS = config.OPTIMIZATION_PARAMS

def _run_single_backtest(symbol, param_dict, start_date, end_date):
    """
    Helper function to run a single backtest for parallel execution.
    """
    orb_timeframe = param_dict.get('ORB_TIMEFRAME', config.ORB_TIMEFRAME)

    performance, _ = run_backtest(
        symbol, 
        start_date, 
        end_date, 
        generate_plot=False, 
        orb_timeframe=orb_timeframe, 
        initial_equity=config.INITIAL_EQUITY_WALK_FORWARD,
        rr_standard=param_dict['RISK_REWARD_RATIO_STANDARD'],
        rr_reversal=param_dict['RISK_REWARD_RATIO_REVERSAL']
    )

    if performance:
        run_summary = {**param_dict, **performance, 'Symbol': symbol}
        return run_summary
    return None

def run_walk_forward_optimization():
    """
    Runs a walk-forward optimization.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = None
    try:
        output_dir = r'D:\\Microsoft VS Code\\Projects\\2025\\ORB_Outputs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file_path = os.path.join(output_dir, 'ORB_Optimizer_Walk_Forward_output.txt')
        log_file = open(output_file_path, 'w')
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = sys.stdout

        print("--- STARTING WALK-FORWARD OPTIMIZATION ---")
        
        all_wfo_results = []
        current_wfo_start = WFO_START_DATE

        while current_wfo_start + timedelta(days=IS_WINDOW_DAYS + OOS_WINDOW_DAYS) <= WFO_END_DATE:
            is_start = current_wfo_start
            is_end = is_start + timedelta(days=IS_WINDOW_DAYS)
            oos_start = is_end
            oos_end = oos_start + timedelta(days=OOS_WINDOW_DAYS)

            print(f"\n--- WFO Step: IS [{is_start.strftime('%Y-%m-%d')} - {is_end.strftime('%Y-%m-%d')}] | OOS [{oos_start.strftime('%Y-%m-%d')} - {oos_end.strftime('%Y-%m-%d')}] ---")
            
            is_results = []
            for symbol, params in OPTIMIZATION_PARAMS.items():
                param_combinations = list(product(*params.values()))
                tasks = [dict(zip(params.keys(), combo)) for combo in param_combinations]
                
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(_run_single_backtest, symbol, param_dict, is_start, is_end) for param_dict in tasks]
                    symbol_is_results = [future.result() for future in futures if future.result() is not None]
                is_results.extend(symbol_is_results)

            if not is_results:
                current_wfo_start += timedelta(days=STEP_DAYS)
                continue

            is_results_df = pd.DataFrame(is_results)
            best_params_per_symbol = is_results_df.loc[is_results_df.groupby('Symbol')['pnl'].idxmax()]

            for index, row in best_params_per_symbol.iterrows():
                symbol = row['Symbol']
                best_param_dict = {k: row[k] for k in OPTIMIZATION_PARAMS[symbol].keys()}
                
                oos_summary = _run_single_backtest(symbol, best_param_dict, oos_start, oos_end)
                if oos_summary:
                    oos_summary['WFO_IS_Start'] = is_start
                    oos_summary['WFO_IS_End'] = is_end
                    oos_summary['WFO_OOS_Start'] = oos_start
                    oos_summary['WFO_OOS_End'] = oos_end
                    all_wfo_results.append(oos_summary)
            
            current_wfo_start += timedelta(days=STEP_DAYS)

        if not all_wfo_results:
            print("No results generated during walk-forward optimization.")
        else:
            final_wfo_df = pd.DataFrame(all_wfo_results)
            output_path = os.path.join(output_dir, "ORB_walk_forward_results.csv")
            final_wfo_df.to_csv(output_path, index=False)
            print("\n--- WALK-FORWARD OPTIMIZATION COMPLETE ---")
            print(f"Walk-forward results saved to {output_path}")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_file:
            log_file.close()

if __name__ == "__main__":
    run_walk_forward_optimization()
