# -*- coding: utf-8 -*-
"""
Optimization framework for the ORB Trading Bot.
"""

import pandas as pd
from itertools import product
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import pytz
import sys
import os

OPTIMIZER_START_DATE = datetime(2024, 1, 1, tzinfo=pytz.UTC)
OPTIMIZER_END_DATE = datetime(2025, 12, 31, tzinfo=pytz.UTC)

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

# --- OPTIMIZATION CONFIGURATION ---
OPTIMIZATION_PARAMS = config.OPTIMIZATION_PARAMS

def analyze_and_suggest(df):
    """Analyzes optimization results to provide data-driven suggestions and rankings."""
    df['drawdown_score'] = 100 * (1 - (abs(df['max_drawdown_pct']) / 100) ** 0.5)
    df['sharpe_score'] = (df['sharpe_ratio'].clip(0, 3) / 3) * 100
    df['profit_score'] = (df['pnl'] > 0).astype(int) * 100

    weights = {'drawdown': 0.4, 'sharpe': 0.3, 'profit': 0.3}
    
    df['Overall Score'] = (
        df['drawdown_score'] * weights['drawdown'] + 
        df['sharpe_score'] * weights['sharpe'] + 
        df['profit_score'] * weights['profit']
    )

    df['Rank'] = df.groupby('Symbol')['Overall Score'].rank(method='dense', ascending=False).astype(int)
    return df

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
        initial_equity=config.INITIAL_EQUITY_OPTIMIZER,
        rr_standard=param_dict['RISK_REWARD_RATIO_STANDARD'],
        rr_reversal=param_dict['RISK_REWARD_RATIO_REVERSAL']
    )

    if performance:
        run_summary = {**param_dict, **performance, 'Symbol': symbol}
        return run_summary
    return None

def run_optimization():
    """
    Runs a grid search optimization.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = None
    try:
        output_dir = r'D:\Microsoft VS Code\Projects\2025\ORB_Outputs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file_path = os.path.join(output_dir, 'ORB_Optimizer_output.txt')
        log_file = open(output_file_path, 'w')
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = sys.stdout

        print("--- STARTING PARAMETER OPTIMIZATION ---")
        
        start_date = OPTIMIZER_START_DATE
        end_date = OPTIMIZER_END_DATE

        all_results = []
        for symbol, params in OPTIMIZATION_PARAMS.items():
            param_combinations = list(product(*params.values()))
            tasks = [dict(zip(params.keys(), combo)) for combo in param_combinations]
            
            print(f"\n--- Optimizing Symbol: {symbol} ({len(tasks)} combinations) ---")

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(_run_single_backtest, symbol, param_dict, start_date, end_date) for param_dict in tasks]
                symbol_results = [future.result() for future in futures if future.result() is not None]
            
            all_results.extend(symbol_results)

        if not all_results:
            print("No results generated during optimization.")
            return

        results_df = pd.DataFrame(all_results)
        final_df = analyze_and_suggest(results_df)
        final_df = final_df.sort_values(by=['Symbol', 'Rank'])

        output_path = os.path.join(output_dir, "ORB_optimization_results.xlsx")
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            final_df.to_excel(writer, sheet_name='Full Optimization Results', index=False)
        
        print("\n--- OPTIMIZATION COMPLETE ---")
        print(f"Optimization results saved to {output_path}")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_file:
            log_file.close()


if __name__ == "__main__":
    run_optimization()
