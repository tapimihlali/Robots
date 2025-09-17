import pandas as pd
from datetime import datetime
import pytz

import MetaTrader5 as mt5
import settings

def analyze_invalidated_trades():
    """
    Performs a what-if analysis on trades closed due to invalidation.
    """
    print("--- STARTING INVALIDATION ANALYSIS ---")

    import pandas as pd
from datetime import datetime
import pytz
import os

import MetaTrader5 as mt5
import settings
import config

def analyze_invalidated_trades():
    """
    Performs a what-if analysis on trades closed due to invalidation.
    """
    print("--- STARTING INVALIDATION ANALYSIS ---")

    output_dir = config.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    backtest_results_path = os.path.join(output_dir, "ICT_backtest_results.csv")

    try:
        results_df = pd.read_csv(backtest_results_path)
    except FileNotFoundError:
        print(f"Error: {backtest_results_path} not found. Please run a backtest first.")
        return

    invalidated_trades = results_df[results_df['reason'] == 'Invalidation'].copy()

    if invalidated_trades.empty:
        print("No trades with 'Invalidation' reason found.")
        return

    print(f"Found {len(invalidated_trades)} invalidated trades to analyze.")

    MT5_USER, MT5_PASS, MT5_SERVER, MT5_PATH, _, _ = settings.synthetic()
    if not mt5.initialize(path=MT5_PATH, login=MT5_USER, password=MT5_PASS, server=MT5_SERVER):
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return

    analysis_results = []
    symbol_infos = {}

    for index, trade in invalidated_trades.iterrows():
        symbol = trade['symbol']
        if symbol not in symbol_infos:
            symbol_infos[symbol] = mt5.symbol_info(symbol)
        
        close_time = pd.to_datetime(trade['close_time'])
        end_of_data = datetime(2025, 8, 29, tzinfo=pytz.UTC)

        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, close_time, end_of_data)
        if rates is None or len(rates) == 0:
            print(f"Could not fetch M1 data for {symbol} after {close_time}. Skipping trade {trade['ticket']}.")
            continue

        future_data = pd.DataFrame(rates)
        future_data['time'] = pd.to_datetime(future_data['time'], unit='s').dt.tz_localize('UTC')

        outcome = "Neither"
        outcome_price = None
        outcome_time = None
        potential_pl = 0

        for _, row in future_data.iterrows():
            if trade['type'] == 'BUY':
                if row['low'] <= trade['sl']:
                    outcome = "Would have hit SL"
                    outcome_price = trade['sl']
                    outcome_time = row['time']
                    break
                elif row['high'] >= trade['tp']:
                    outcome = "Would have hit TP"
                    outcome_price = trade['tp']
                    outcome_time = row['time']
                    break
            elif trade['type'] == 'SELL':
                if row['high'] >= trade['sl']:
                    outcome = "Would have hit SL"
                    outcome_price = trade['sl']
                    outcome_time = row['time']
                    break
                elif row['low'] <= trade['tp']:
                    outcome = "Would have hit TP"
                    outcome_price = trade['tp']
                    outcome_time = row['time']
                    break
        
        symbol_info = symbol_infos[symbol]
        point = symbol_info.point
        tick_value = symbol_info.trade_tick_value
        volume = trade['volume']

        if outcome == "Would have hit TP":
            pips = (trade['tp'] - trade['entry_price']) / point if trade['type'] == 'BUY' else (trade['entry_price'] - trade['tp']) / point
            potential_pl = pips * tick_value * volume
        elif outcome == "Would have hit SL":
            pips = (trade['sl'] - trade['entry_price']) / point if trade['type'] == 'BUY' else (trade['entry_price'] - trade['sl']) / point
            potential_pl = pips * tick_value * volume

        trade_result = trade.to_dict()
        trade_result['What-If Outcome'] = outcome
        trade_result['Outcome Price'] = outcome_price
        trade_result['Outcome Time'] = outcome_time
        trade_result['Potential P/L'] = potential_pl
        analysis_results.append(trade_result)

    mt5.shutdown()

    if analysis_results:
        analysis_df = pd.DataFrame(analysis_results)
        
        if 'open_time' in analysis_df.columns: analysis_df['open_time'] = pd.to_datetime(analysis_df['open_time']).dt.tz_localize(None)
        if 'close_time' in analysis_df.columns: analysis_df['close_time'] = pd.to_datetime(analysis_df['close_time']).dt.tz_localize(None)
        if 'Outcome Time' in analysis_df.columns: analysis_df['Outcome Time'] = pd.to_datetime(analysis_df['Outcome Time']).dt.tz_localize(None)

        # --- Generate Summaries ---
        symbol_summary = analysis_df.groupby('symbol').apply(lambda x: pd.Series({
            'Total Invalidated': len(x),
            'Would Hit TP': (x['What-If Outcome'] == 'Would have hit TP').sum(),
            'Would Hit SL': (x['What-If Outcome'] == 'Would have hit SL').sum(),
            'Would Hit Neither': (x['What-If Outcome'] == 'Neither').sum(),
            'Potential P/L if TP': x[x['What-If Outcome'] == 'Would have hit TP']['Potential P/L'].sum(),
            'Potential P/L if SL': x[x['What-If Outcome'] == 'Would have hit SL']['Potential P/L'].sum(),
        })).reset_index()

        total_summary = pd.DataFrame([{ 
            'Total Invalidated': len(analysis_df),
            'Would Hit TP': (analysis_df['What-If Outcome'] == 'Would have hit TP').sum(),
            'Would Hit SL': (analysis_df['What-If Outcome'] == 'Would have hit SL').sum(),
            'Would Hit Neither': (analysis_df['What-If Outcome'] == 'Neither').sum(),
            'Potential P/L if TP': analysis_df[analysis_df['What-If Outcome'] == 'Would have hit TP']['Potential P/L'].sum(),
            'Potential P/L if SL': analysis_df[analysis_df['What-If Outcome'] == 'Would have hit SL']['Potential P/L'].sum(),
        }])

        excel_output_path = os.path.join(output_dir, "ICT_invalidation_analysis.xlsx")
        with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
            analysis_df.to_excel(writer, sheet_name="Invalidation Analysis", index=False)
            symbol_summary.to_excel(writer, sheet_name="Symbol Summary", index=False)
            total_summary.to_excel(writer, sheet_name="Total Summary", index=False)

        print("\n--- ANALYSIS COMPLETE ---")
        print(f"Invalidation analysis saved to {excel_output_path}")
    else:
        print("\n--- ANALYSIS COMPLETE ---")
        print("No analysis results were generated.")

if __name__ == "__main__":
    analyze_invalidated_trades()

    invalidated_trades = results_df[results_df['reason'] == 'Invalidation'].copy()

    if invalidated_trades.empty:
        print("No trades with 'Invalidation' reason found.")
        return

    print(f"Found {len(invalidated_trades)} invalidated trades to analyze.")

    MT5_USER, MT5_PASS, MT5_SERVER, MT5_PATH, _, _ = settings.synthetic()
    if not mt5.initialize(path=MT5_PATH, login=MT5_USER, password=MT5_PASS, server=MT5_SERVER):
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return

    analysis_results = []
    symbol_infos = {}

    for index, trade in invalidated_trades.iterrows():
        symbol = trade['symbol']
        if symbol not in symbol_infos:
            symbol_infos[symbol] = mt5.symbol_info(symbol)
        
        close_time = pd.to_datetime(trade['close_time'])
        end_of_data = datetime(2025, 8, 29, tzinfo=pytz.UTC)

        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, close_time, end_of_data)
        if rates is None or len(rates) == 0:
            print(f"Could not fetch M1 data for {symbol} after {close_time}. Skipping trade {trade['ticket']}.")
            continue

        future_data = pd.DataFrame(rates)
        future_data['time'] = pd.to_datetime(future_data['time'], unit='s').dt.tz_localize('UTC')

        outcome = "Neither"
        outcome_price = None
        outcome_time = None
        potential_pl = 0

        for _, row in future_data.iterrows():
            if trade['type'] == 'BUY':
                if row['low'] <= trade['sl']:
                    outcome = "Would have hit SL"
                    outcome_price = trade['sl']
                    outcome_time = row['time']
                    break
                elif row['high'] >= trade['tp']:
                    outcome = "Would have hit TP"
                    outcome_price = trade['tp']
                    outcome_time = row['time']
                    break
            elif trade['type'] == 'SELL':
                if row['high'] >= trade['sl']:
                    outcome = "Would have hit SL"
                    outcome_price = trade['sl']
                    outcome_time = row['time']
                    break
                elif row['low'] <= trade['tp']:
                    outcome = "Would have hit TP"
                    outcome_price = trade['tp']
                    outcome_time = row['time']
                    break
        
        symbol_info = symbol_infos[symbol]
        point = symbol_info.point
        tick_value = symbol_info.trade_tick_value
        volume = trade['volume']

        if outcome == "Would have hit TP":
            pips = (trade['tp'] - trade['entry_price']) / point if trade['type'] == 'BUY' else (trade['entry_price'] - trade['tp']) / point
            potential_pl = pips * tick_value * volume
        elif outcome == "Would have hit SL":
            pips = (trade['sl'] - trade['entry_price']) / point if trade['type'] == 'BUY' else (trade['entry_price'] - trade['sl']) / point
            potential_pl = pips * tick_value * volume

        trade_result = trade.to_dict()
        trade_result['What-If Outcome'] = outcome
        trade_result['Outcome Price'] = outcome_price
        trade_result['Outcome Time'] = outcome_time
        trade_result['Potential P/L'] = potential_pl
        analysis_results.append(trade_result)

    mt5.shutdown()

    if analysis_results:
        analysis_df = pd.DataFrame(analysis_results)
        
        if 'open_time' in analysis_df.columns: analysis_df['open_time'] = pd.to_datetime(analysis_df['open_time']).dt.tz_localize(None)
        if 'close_time' in analysis_df.columns: analysis_df['close_time'] = pd.to_datetime(analysis_df['close_time']).dt.tz_localize(None)
        if 'Outcome Time' in analysis_df.columns: analysis_df['Outcome Time'] = pd.to_datetime(analysis_df['Outcome Time']).dt.tz_localize(None)

        # --- Generate Summaries ---
        symbol_summary = analysis_df.groupby('symbol').apply(lambda x: pd.Series({
            'Total Invalidated': len(x),
            'Would Hit TP': (x['What-If Outcome'] == 'Would have hit TP').sum(),
            'Would Hit SL': (x['What-If Outcome'] == 'Would have hit SL').sum(),
            'Would Hit Neither': (x['What-If Outcome'] == 'Neither').sum(),
            'Potential P/L if TP': x[x['What-If Outcome'] == 'Would have hit TP']['Potential P/L'].sum(),
            'Potential P/L if SL': x[x['What-If Outcome'] == 'Would have hit SL']['Potential P/L'].sum(),
        })).reset_index()

        total_summary = pd.DataFrame([{
            'Total Invalidated': len(analysis_df),
            'Would Hit TP': (analysis_df['What-If Outcome'] == 'Would have hit TP').sum(),
            'Would Hit SL': (analysis_df['What-If Outcome'] == 'Would have hit SL').sum(),
            'Would Hit Neither': (analysis_df['What-If Outcome'] == 'Neither').sum(),
            'Potential P/L if TP': analysis_df[analysis_df['What-If Outcome'] == 'Would have hit TP']['Potential P/L'].sum(),
            'Potential P/L if SL': analysis_df[analysis_df['What-If Outcome'] == 'Would have hit SL']['Potential P/L'].sum(),
        }])

        excel_output_path = os.path.join(output_dir, "ICT_invalidation_analysis.xlsx")
        with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
            analysis_df.to_excel(writer, sheet_name="Invalidation Analysis", index=False)
            symbol_summary.to_excel(writer, sheet_name="Symbol Summary", index=False)
            total_summary.to_excel(writer, sheet_name="Total Summary", index=False)

        print("\n--- ANALYSIS COMPLETE ---")
        print("Invalidation analysis saved to ICT_invalidation_analysis.xlsx")
    else:
        print("\n--- ANALYSIS COMPLETE ---")
        print("No analysis results were generated.")

if __name__ == "__main__":
    analyze_invalidated_trades()
