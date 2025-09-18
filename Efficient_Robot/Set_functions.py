import pandas as pd
import numpy as np
import datetime as dt
import pytz
# from instrument import Instrument
from scipy.signal import savgol_filter, argrelextrema, find_peaks
from scipy import signal

import pywt

import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import MetaTrader5 as mt5

import indicators as indie

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def granular(grain='M1'):
    
    timeframes_mapping = {
    'M1': [mt5.TIMEFRAME_M1, 1],
    'M2': [mt5.TIMEFRAME_M2, 2],
    'M3': [mt5.TIMEFRAME_M3, 3],
    'M4': [mt5.TIMEFRAME_M4, 4],
    'M5': [mt5.TIMEFRAME_M5, 5],
    'M6': [mt5.TIMEFRAME_M6, 6],
    'M10': [mt5.TIMEFRAME_M10, 10],
    'M12': [mt5.TIMEFRAME_M12, 12],
    'M15': [mt5.TIMEFRAME_M15, 15],
    'M30': [mt5.TIMEFRAME_M30, 30],
    'H1': [mt5.TIMEFRAME_H1, 60],
    'H2': [mt5.TIMEFRAME_H2, 120],
    'H3': [mt5.TIMEFRAME_H3, 180],
    'H4': [mt5.TIMEFRAME_H4, 240],
    'H6': [mt5.TIMEFRAME_H6, 360],
    'H8': [mt5.TIMEFRAME_H8, 480],
    'H12': [mt5.TIMEFRAME_H12, 720],
    'D1': [mt5.TIMEFRAME_D1, 1440]    
    }

    return timeframes_mapping[grain][0]

def fetch_candles(pair_name,  start_date, end_date, granularity, price):       
    grain = granular(granularity)
    
    if price == 'Price':
        price = mt5.copy_rates_range(pair_name, grain, start_date, end_date)
        price = pd.DataFrame(price)
        price['time'] = pd.to_datetime(price['time'], unit='s')
        price = price.rename(columns={'time': 'date', 'tick_volume': 'volume'})
        #price = price.set_index('date')

    elif price == 'Tick':
        price = mt5.copy_ticks_range(pair_name, start_date, end_date, mt5.COPY_TICKS_ALL)
        price = pd.DataFrame(price)
        price['date'] = pd.to_datetime(price['time_msc'], unit='ms')
        price.set_index('date', inplace=True)
        price.drop('time', axis=1, inplace=True)
        price.drop('time_msc', axis=1, inplace=True)
        price = price[price.columns[price.any()]]
        #price = price.set_index('date')   
    return price

def clean_tick_data(df: pd.DataFrame,
                    n_digits: int,
                    timezone: str = 'Etc/GMT-2'
                    ):
    """
    Clean and validate Forex tick data with comprehensive quality checks.

    Args:
        df: DataFrame containing tick data with bid/ask prices and timestamp index
        n_digits: Number of decimal places in instrument price.
        timezone: Timezone to localize/convert timestamps to (default: 'Etc/GMT-2')

    Returns:
        Cleaned DataFrame or None if empty after cleaning
    """
    if df.empty:
        return None

    df = df.copy(deep=False)  # Work on a copy to avoid modifying the original DataFrame 
    n_initial = df.shape[0] # Store initial row count for reporting

    # 1. Ensure proper datetime index
    # Use errors='coerce' to turn unparseable dates into NaT and then drop them.
    if not isinstance(df.index, pd.DatetimeIndex):
        original_index_name = df.index.name
        df.index = pd.to_datetime(df.index, errors='coerce')
        nan_idx_count = df.index.isnull().sum()
        if nan_idx_count > 0:
            logging.info(f"Dropped {nan_idx_count:,} rows with unparseable timestamps.")
            df = df[~df.index.isnull()]
        if original_index_name:
            df.index.name = original_index_name
    
    if df.empty: # Check if empty after index cleaning
        logging.warning("Warning: DataFrame empty after initial index cleaning")
        return None

    # 2. Timezone handling
    if df.index.tz is None:
        df = df.tz_localize(timezone)
    elif str(df.index.tz) != timezone.upper():
        df = df.tz_convert(timezone)
    
    # 3. Price validity checks
    # Apply rounding and then filtering
    df['bid'] = df['bid'].round(n_digits)
    df['ask'] = df['ask'].round(n_digits)

    # Validate prices
    price_filter = (
        (df['bid'] > 0) &
        (df['ask'] > 0) &
        (df['ask'] > df['bid'])
    )
    
    n_before_price_filter = df.shape[0]
    df = df[price_filter]
    n_filtered_prices = n_before_price_filter - df.shape[0]
    if n_filtered_prices > 0:
        logging.info(f"Filtered {n_filtered_prices:,} ({n_filtered_prices / n_before_price_filter:.2%}) invalid prices.")

    if df.empty: # Check if empty after price cleaning
        logging.warning("Warning: DataFrame empty after price cleaning")
        return None
    
    # Dropping NA values
    initial_rows_before_na = df.shape[0]
    if df.isna().any().any(): # Use .any().any() to check if any NA exists in the whole DF
        na_counts = df.isna().sum()
        na_cols = na_counts[na_counts > 0]
        if not na_cols.empty:
            logging.info(f'Dropped NA values from columns: \n{na_cols}')
            df.dropna(inplace=True)

    n_dropped_na = initial_rows_before_na - df.shape[0]
    if n_dropped_na > 0:
        logging.info(f"Dropped {n_dropped_na:,} ({n_dropped_na / n_before_price_filter:.2%}) rows due to NA values.")

    if df.empty: # Check if empty after NA cleaning
        logging.warning("Warning: DataFrame empty after NA cleaning")
        return None
    
    # 4. Microsecond handling
    if not df.index.microsecond.any():
        logging.warning("Warning: No timestamps with microsecond precision found")
    
    # 5. Duplicate handling
    duplicate_mask = df.index.duplicated(keep='last')
    dup_count = duplicate_mask.sum()
    if dup_count > 0:
        logging.info(f"Removed {dup_count:,} ({dup_count / n_before_price_filter:.2%}) duplicate timestamps.")
        df = df[~duplicate_mask]

    if df.empty: # Check if empty after duplicate cleaning
        logging.warning("Warning: DataFrame empty after duplicate cleaning")
        return None

    # 6. Chronological order
    if not df.index.is_monotonic_increasing:
        logging.info("Sorting DataFrame by index to ensure chronological order.")
        df.sort_index(inplace=True)

    # 7. Final validation and reporting
    if df.empty:
        logging.warning("Warning: DataFrame empty after all cleaning steps.")
        return None
    
    n_final = df.shape[0]
    n_cleaned = n_initial - n_final
    percentage_cleaned = (n_cleaned / n_initial) if n_initial > 0 else 0
    logging.info(f"Cleaned {n_cleaned:,} of {n_initial:,} ({percentage_cleaned:.2%}) datapoints.")

    return df

def set_resampling_freq(timeframe: str) -> str:
    """
    Converts an MT5 timeframe to a pandas resampling frequency.

    Args:
        timeframe (str): MT5 timeframe (e.g., 'M1', 'M5', 'M15', 'H1', 'H4', 'D1', 'W1').

    Returns:
        str: Pandas frequency string.
    """
    timeframe = timeframe.upper()
    nums = [x for x in timeframe if x.isnumeric()]
    if not nums:
        raise ValueError("Timeframe must include numeric values (e.g., 'M1').")
    
    x = int(''.join(nums))
    if timeframe == 'W1':
        freq = 'W-FRI'
    elif timeframe == 'D1':
        freq = 'B'
    elif timeframe.startswith('H'):
        freq = f'{x}H'
    elif timeframe.startswith('M'):
        freq = f'{x}min'
    elif timeframe.startswith('S'):
        freq = f'{x}S'
    else:
        raise ValueError("Valid timeframes include W1, D1, Hx, Mx, Sx.")
    
    return freq

def calculate_ticks_per_period(df: pd.DataFrame, timeframe: str = "M1", method: str = 'median', verbose: bool = True) -> int:
    """
    Dynamically calculates the average number of ticks per given timeframe.

    Args:
        df (pd.DataFrame): Tick data.
        timeframe (str): MT5 timeframe.
        method (str): 'median' or 'mean' for the calculation.
        verbose (bool): Whether to print the result.

    Returns:
        int: Rounded average ticks per period.
    """
    freq = set_resampling_freq(timeframe)
    resampled = df.resample(freq).size()
    fn = getattr(np, method)
    num_ticks = fn(resampled.values)
    num_rounded = int(np.round(num_ticks))
    num_digits = len(str(num_rounded)) - 1
    rounded_ticks = int(round(num_rounded, -num_digits))
    rounded_ticks = max(1, rounded_ticks)
    
    if verbose:
        t0 = df.index[0].date()
        t1 = df.index[-1].date()
        logging.info(f"From {t0} to {t1}, {method} ticks per {timeframe}: {num_ticks:,} rounded to {rounded_ticks:,}")
    
    return rounded_ticks

def flatten_column_names(df):
    '''
    Joins tuples created by dataframe aggregation 
    with a list of functions into a unified name.
    '''
    return ["_".join(col).strip() for col in df.columns.values]

def make_bar_type_grouper(
        df: pd.DataFrame,
        bar_type: str = 'tick',
        bar_size: int = 100,
        timeframe: str = 'M1'
) -> tuple[pd.core.groupby.generic.DataFrameGroupBy, int]:
    """
    Create a grouped object for aggregating tick data into time/tick/dollar/volume bars.

    Args:
        df: DataFrame with tick data (index should be datetime for time bars).
        bar_type: Type of bar ('time', 'tick', 'dollar', 'volume').
        bar_size: Number of ticks/dollars/volume per bar (ignored for time bars).
        timeframe: Timeframe for resampling (e.g., 'h1', 'D1', 'W1').

    Returns:
        - GroupBy object for aggregation
        - Calculated bar_size (for tick/dollar/volume bars)
    """
    # Create working copy (shallow is sufficient)
    df = df.copy(deep=False)  # OPTIMIZATION: Shallow copy here only once
    
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.set_index('time')
        except KeyError:
            raise TypeError("Could not set 'time' as index")

    # Sort if needed
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # Time bars
    if bar_type == 'time':
        
        if timeframe == 'H1':
            timeframe = 'h1'
            
        freq = set_resampling_freq(timeframe)
        bar_group = (df.resample(freq, closed='left', label='right') # includes data upto, but not including, the end of the period
                    if not freq.startswith(('B', 'W')) 
                    else df.resample(freq))
        return bar_group, 0  # bar_size not used

    # Dynamic bar sizing
    if bar_size == 0:
        if bar_type == 'tick':
            bar_size = calculate_ticks_per_period(df, timeframe)
        else:
            raise NotImplementedError(f"{bar_type} bars require non-zero bar_size")

    # Non-time bars
    df['time'] = df.index  # Add without copying
    
    if bar_type == 'tick':
        bar_id = np.arange(len(df)) // bar_size
    elif bar_type in ('volume', 'dollar'):
        if 'volume' not in df.columns:
            raise KeyError(f"'volume' column required for {bar_type} bars")
        
        # Optimized cumulative sum
        cum_metric = (df['volume'] * df['bid'] if bar_type == 'dollar' 
                      else df['volume'])
        cumsum = cum_metric.cumsum()
        bar_id = (cumsum // bar_size).astype(int)
    else:
        raise NotImplementedError(f"{bar_type} bars not implemented")

    return df.groupby(bar_id), bar_size

def make_bars(tick_df: pd.DataFrame,
              bar_type: str = 'tick',
              bar_size: int = 0,
              timeframe: str = 'M1',
              price: str = 'midprice',
              verbose=True):
    '''
    Create OHLC data by sampling ticks using timeframe or a threshold.

    Parameters
    ----------
    tick_df: pd.DataFrame
        tick data
    bar_type: str
        type of bars to create from ['tick', 'time', 'volume', 'dollar']
    bar_size: int 
        default 0. bar_size when bar_type != 'time'
    timeframe: str
        MT5 timeframe (e.g., 'M5', 'H1', 'D1', 'W1').
        Used for time bars, or for tick bars if bar_size = 0.
    price: str
        default midprice. If 'bid_ask', columns (bid_open, ..., bid_close), 
        (ask_open, ..., ask_close) are included.
    verbose: bool
        print information about the data

    Returns
    -------
    pd.DataFrame with columns [open, high, low, close, median_price, tick_volume, volume]
    '''    
    if 'midprice' not in tick_df:
        tick_df['midprice'] = (tick_df['bid'] + tick_df['ask']) / 2

    bar_group, bar_size_ = make_bar_type_grouper(tick_df, bar_type, bar_size, timeframe)
    ohlc_df = bar_group['midprice'].ohlc().astype('float64')
    ohlc_df['tick_volume'] = bar_group['bid'].count() if bar_type != 'tick' else bar_size_
    
    if price == 'bid_ask':
        # Aggregate OHLC data for every bar_size rows
        bid_ask_df = bar_group.agg({k: 'ohlc' for k in ('bid', 'ask')})
        # Flatten MultiIndex columns
        col_names = flatten_column_names(bid_ask_df)
        bid_ask_df.columns = col_names
        ohlc_df = ohlc_df.join(bid_ask_df)

    if 'volume' in tick_df:
        ohlc_df['volume'] = bar_group['volume'].sum()

    if bar_type == 'time':
        ohlc_df.ffill(inplace=True)
    else:
        end_time =  bar_group['time'].last()
        ohlc_df.index = end_time + pd.Timedelta(microseconds=1) # ensure end time is after event
	    # ohlc_df.drop('time', axis=1, inplace=True) # Remove 'time' column


        # drop last bar due to insufficient ticks
        if len(tick_df) % bar_size_ > 0: 
            ohlc_df = ohlc_df.iloc[:-1]

    if verbose:
        if bar_type != 'time':
            tm = f'{bar_size_:,}'
            if bar_type == 'tick' and bar_size == 0:
                tm = f'{timeframe} - {bar_size_:,} ticks'
            timeframe = tm
        print(f'\nTick data - {tick_df.shape[0]:,} rows')
        print(f'{bar_type}_bar {timeframe}')
        ohlc_df.info()
    
    # # Remove timezone info from DatetimeIndex
    # try:
    # ohlc_df = ohlc_df.tz_convert(None)
     
    # except:
	#     pass
    
    return ohlc_df

def get_tick(pair, days=5, granularity='H1', timezone='Etc/GMT-2', price='Tick'):
    
    tick = collect_price(pair=pair, days=days, granularity=granularity, timezone=timezone, price=price)
    n_dig = mt5.symbols_get(pair)[0].digits
    # tick_01 = clean_tick_data(df=tick, n_digits=n_dig)
    tick_02 = make_bars(tick_df=tick,bar_type='time',bar_size=2,timeframe=granularity,price='midprice',verbose=False)
    
    return tick_02

# Functions Concerned with getting Price data
def collect_price(pair, days=5, granularity='M1',  timezone='Etc/GMT-2', price='Tick'):    
    zone = pytz.timezone(timezone)
    now = dt.datetime.now(tz=zone)
    dayz = days
    start_date = now - dt.timedelta(days=dayz)

    if start_date.weekday() == 6:
        start_date -=dt.timedelta(days=2)

    elif start_date.weekday() == 5:
        start_date -=dt.timedelta(days=1)

    date_from = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    date_to = dt.datetime(now.year, now.month, now.day, now.hour, now.minute, now.second)
    
    # collect candles
    # ap = data_api.DataApi()
    price = fetch_candles(pair,granularity=granularity,start_date=date_from,end_date=date_to,price=price)
    
    if len(price) > 0:
        final_df = price
    elif len(price) == 0:
        print('ERROR', pair, granularity, date_from, date_to)
        
    # print(f'{pair} {granularity} {final_df.iloc[0].date} {final_df.iloc[-1].date}' )
    return final_df

# # Functions Concerned with getting Price data
# def collect_price(pair, api='Synthetic', days=5, granularity='M1',  timezone='Etc/GMT-2', price='Price'):    
#     zone = pytz.timezone(timezone)
#     now = dt.datetime.now(tz=zone)
#     dayz = days
#     start_date = now - dt.timedelta(days=dayz)

#     if start_date.weekday() == 6:
#         start_date -=dt.timedelta(days=2)

#     elif start_date.weekday() == 5:
#         start_date -=dt.timedelta(days=1)

#     date_from = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
#     date_to = dt.datetime(now.year, now.month, now.day, now.hour, now.minute, now.second)
    
#     # collect candles
#     if api=='Forex':
#         ap = data_api.DataApi()
#         price = ap.fetch_candles(pair,granularity=granularity,start_date=date_from,end_date=date_to,price=price)
        
#     elif api=='Synthetic':
#         ap = syn_api.DataApi()
#         price = ap.fetch_candles(pair,granularity=granularity,start_date=date_from,end_date=date_to,price=price)
    
#     if len(price) > 0:
#         final_df = price
#     elif len(price) == 0:
#         print('ERROR', pair, granularity, date_from, date_to)
        
#     # print(f'{pair} {granularity} {final_df.iloc[0].date} {final_df.iloc[-1].date}' )
#     return final_df

# def get_symbols(api='Synthetic'):
#     if api=='Forex':
#         ap = data_api.DataApi()
#         Sym_list = ap.fetch_instruments()
        
#         # create_list = []
#         # for i in range(len(Sym_list)):
#         #     create_list.append(Sym_list[i].name)
#         # create_list
        
#     elif api=='Synthetic':
#         ap = syn_api.DataApi()
#         Sym_list = ap.fetch_instruments()
        
#         # create_list = []
#         # for i in range(len(Sym_list)):
#         #     create_list.append(Sym_list[i].name)
#         # create_list
    
#     return Sym_list

# Functions for getting market symbols and instruments
def get_instruments_df():
    instrument_data = []
    ins = mt5.symbols_get()
    for item in range(len(ins)):
        new_ob = dict(
            name = ins[item].name,
            type = ins[item].path,
            displayName = ins[item].description,
            pipLocation = ins[item].digits,
            marginRate = ins[item].volume_min
        )
        instrument_data.append(new_ob)
    instrument_df = pd.DataFrame.from_dict(instrument_data)
    return instrument_df

def market_types(Synth_list):    
    lis = []
    for i in range(len(Synth_list)):
        j = 0

        lend = ""
        while Synth_list[i].path[j] != "\\":
            
            lend = f'{lend}'+Synth_list[i].path[j]
            j += 1
        lis.append(lend)

    condensed = list(set(lis))

    secondary = []
    for j in range(len(condensed)):
        initial = []
        for i in range(len(lis)):
            if lis[i] == condensed[j]:
                initial.append(Synth_list[i].name)
        secondary.append(initial)
    
    return condensed, secondary

def update_loss_count(symbol, profit):
    """
    Updates the loss count for a symbol based on the trade outcome.
    If the trade is a loss, increment the count; if it's a win, reset it.
    """
    global loss_count

    if profit < 0:  # Trade closed at a loss
        loss_count[symbol] = loss_count.get(symbol, 0) + 1
    else:  # Trade closed at a profit
        loss_count[symbol] = 0  # Reset count

    print(f"Updated loss count: {loss_count}")
    
def calculate_lot_size(symbol, BASE_LOT_SIZE, MAX_LOT_SIZE=1):
    """
    Calculate the lot size based on the number of consecutive losses.
    If the lot size exceeds the max allowed, cap it.
    """
    loss_multiplier = loss_count.get(symbol, 0)
    new_lot_size = BASE_LOT_SIZE * (1 + loss_multiplier)

    # Ensure the lot size doesn't exceed the max limit
    # return min(new_lot_size, MAX_LOT_SIZE)
    return new_lot_size

def check_closed_trades():
    """
    Check for closed trades and update the loss count.
    """
        # Get the history of trades
    from_date = pd.Timestamp.now().replace(hour=0, minute=0, second=0)  # Start of the day
    to_date = pd.Timestamp.now()  # Current time
    history = mt5.history_deals_get(from_date, to_date)

    if history is None:
        print("No trade history found.")
        return

    for deal in history:
        symbol = deal.symbol
        profit = deal.profit

        # Update loss count
        update_loss_count(symbol, profit)
        
def Positions_df(positions):
    direction = []
    entry_price = []
    entry_time = []
    entry_symbol = []

    # for i in range(len(positions)):
    direction.append(positions.type) # direction
    entry_price.append(positions.price_open) # Entry price
    entry_time.append(pd.to_datetime(positions.time, unit='s')) # Entry time
    entry_symbol.append(positions.symbol)
        
    Positions_dataframe = pd.DataFrame()
    Positions_dataframe['Symbol'] = entry_symbol
    Positions_dataframe['Direction'] = direction
    Positions_dataframe['Entry Price'] = entry_price
    Positions_dataframe['Entry Time'] = entry_time
    Positions_dataframe = Positions_dataframe.set_index('Entry Time')
    
    return Positions_dataframe

def entry_positions(Positions_dataframe, symbol, super_swing, multiplier):
        
    Entries_df = Positions_dataframe
    Exit_loss = []

    for i in range(len(super_swing)):
        for j in range(len(Entries_df)):
            # if symbol == Entries_df.Symbol.iloc[j]:
                if super_swing['Start Date'].iloc[i] < Entries_df.index[j] and super_swing['End Date'].iloc[i] > Entries_df.index[j]:
                    
                    if super_swing['Swing Type'].iloc[i] == 'up' and Entries_df['Direction'].iloc[j] == 1:  
                        Exit_loss.append(super_swing['Highest Close'].iloc[i-2])
                
                    elif super_swing['Swing Type'].iloc[i] == 'down' and Entries_df['Direction'].iloc[j] == 1:  
                        Exit_loss.append(super_swing['Highest Close'].iloc[i-1])
                
                    elif super_swing['Swing Type'].iloc[i] == 'down' and Entries_df['Direction'].iloc[j] == 0:   
                        Exit_loss.append(super_swing['Lowest Close'].iloc[i-2])
                        
                    elif super_swing['Swing Type'].iloc[i] == 'up' and Entries_df['Direction'].iloc[j] == 0:   
                        Exit_loss.append(super_swing['Lowest Close'].iloc[i-1])

    Entries_df['Exit Loss'] = Exit_loss
            
    Entries_df = Entries_df.reset_index()

    tp_sl_result = []

    tp_multiplier = multiplier
    sl_multiplier = multiplier

    for _, entry in Entries_df.iterrows():
        entry_time = entry['Entry Time']
        entry_price = entry['Entry Price']
        entry_loss = entry['Exit Loss']
        direction = entry['Direction']

        # Calculate TP and SL based on the direction
        # sl = np.nan
        # tp = np.nan
        if direction == 0:
            tp = round(entry_price + (tp_multiplier * (entry_price / (1 + tp_multiplier))), 2)
            sl_loose = round(entry_price - (sl_multiplier * (entry_price / (1 + sl_multiplier))), 2)
            
            if sl_loose < entry_loss:
                sl = sl_loose
            
            elif sl_loose > entry_loss:
                sl = round(entry_loss,2)   
            
        elif direction == 1:
            tp = round(entry_price - (tp_multiplier * (entry_price / (1 + tp_multiplier))),2)
            sl_loose = round(entry_price + (sl_multiplier * (entry_price / (1 + sl_multiplier))),2)
            
            if sl_loose > entry_loss:
                sl = sl_loose
            
            elif sl_loose < entry_loss:
                sl = round(entry_loss,2) 

        tp_sl_result.append({'Entry Time': entry_time, 'TP': tp, 'SL': sl})
        
    tp_sl_results = pd.DataFrame(tp_sl_result)
    
    return Entries_df, tp_sl_results

def super_ranges(df, df_0, z_wave):
    # Identify ranges for downward and upward slopes
    ranges = []
    current_range = []
    current_type = None
    
    df_01 = df.rename(columns={'Close': 'close', 'Low': 'low', 'High': 'high', 'Date': 'date', 'Open' : 'open'})
    
    sup = supertrend(df_01, df_upper=False, atr_multiplier=3)
    sup_01 = generate_signals(sup)
    sup_02 = create_positions(sup_01)

    sup_02b = sup_02[-len(z_wave):]
    data = sup_02b

    zero_line = 0
    
    for i, slope in enumerate(data.signals):
        if slope == 1:  # Downward slope
            if current_type != 'down':
                if current_range:
                    ranges.append((current_type, current_range))
                current_range = []
                current_type = 'down'
            current_range.append(i)
        elif slope == 2:  # Upward slope
            if current_type != 'up':
                if current_range:
                    ranges.append((current_type, current_range))
                current_range = []
                current_type = 'up'
            current_range.append(i)
            

    # Append the last range
    if current_range:
        ranges.append((current_type, current_range))
        
    # Analyze swings and classify above/below zero
    super_results = []
    for swing_type, indices in ranges:
        swing_slopes = z_wave.iloc[indices]#data.iloc[indices]['signals']
        
        swing_prices = data.iloc[indices]['close']
        swing_distance = abs(swing_prices.max() - swing_prices.min())
        
        # swing_high = df_.iloc[indices]['High']
        # swing_low = df_.iloc[indices]['Low']
        # swing_distance = abs(swing_high.max() - swing_low.min())
        
        swing_dates = data.index[indices]
        
        # Determine majority above or below zero
        above_count = sum(1 for slope in swing_slopes if slope > zero_line)
        below_count = sum(1 for slope in swing_slopes if slope <= zero_line)
        swing_location = 'above' if above_count > below_count else 'below'
        
        super_results.append({
            'Swing Type': swing_type,
            'Start Date': swing_dates[0],
            'End Date': swing_dates[-1],
            'Swing Distance': swing_distance,
            'Highest Close': swing_prices.max(),
            'Lowest Close': swing_prices.min(),
            'Swing Location': swing_location,
            'Indices': indices
        })

    super_results.append({
            'Swing Type': swing_type,
            'Start Date': swing_dates[0],
            'End Date': df_0.index[-1], #pd.Timestamp.now(),
            'Swing Distance': swing_distance,
            'Highest Close': swing_prices.max(),
            'Lowest Close': swing_prices.min(),
            'Swing Location': swing_location,
            'Indices': indices
        })

    # Create a DataFrame from the results
    super_swing = pd.DataFrame(super_results)
    
    return super_results, super_swing

def close_all_trades(symbol, positions):
    """Close all open trades for a given symbol."""
    for pos in positions:
        if pos.symbol == symbol:
            close_single_pair(symbol, pos)
            print(f"Closed trade {pos.ticket} on {symbol}")
            
def close_all_trades_01(symbol, positions, type=0):
    """Close all open trades for a given symbol."""
    for pos in positions:
        if pos.symbol == symbol and pos.type == type:
            close_single_pair(symbol, pos)
            print(f"Closed trade {pos.ticket} on {symbol}")

def close_single_pair(Symbol, open_position):
        
        symbol = open_position.symbol
        if symbol == Symbol:
            #order_ticket = mt5.positions_get()[i].ticket
            deal_id = open_position.ticket
        
            #open_positions = open_positions[open_positions['ticket'] == deal_id]
            order_type  = open_position.type
            volume = open_position.volume

            if(order_type == mt5.ORDER_TYPE_BUY):
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            
            close_request={
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": order_type,
                "position": deal_id,
                "price": price,
                #"magic": 240000,
                "comment": "Close trade",
                "type_time": mt5.ORDER_TIME_GTC,
                #"type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(close_request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return print("Failed to close order :(")
            else:
                return print (f"Order for {symbol} successfully closed!")

def close_pair(Symbol, open_positions):
        
    for i in range(len(open_positions)):
        symbol = open_positions[i].symbol
        if symbol == Symbol:
            #order_ticket = mt5.positions_get()[i].ticket
            deal_id = open_positions[i].ticket
        
            #open_positions = open_positions[open_positions['ticket'] == deal_id]
            order_type  = open_positions[i].type
            volume = open_positions[i].volume

            if(order_type == mt5.ORDER_TYPE_BUY):
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            
            close_request={
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": order_type,
                "position": deal_id,
                "price": price,
                #"magic": 240000,
                "comment": "Close trade",
                "type_time": mt5.ORDER_TIME_GTC,
                #"type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(close_request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return print("Failed to close order :(")
            else:
                return print (f"Order for {symbol} successfully closed!")

def close_position(open_positions, tp=5, sl=-3):
    
    #for i in range(len(open_positions)):
    Currents=[]
    Profits=[]
    for i in range(len(open_positions)):
        Profits.append(open_positions[i].profit)
        Currents.append(open_positions[i].symbol)
    
    for i in range(len(open_positions)):
        if Profits[i] >= tp or Profits[i] <= sl:
             #order_ticket = mt5.positions_get()[i].ticket
            deal_id = open_positions[i].ticket
        
            #open_positions = open_positions[open_positions['ticket'] == deal_id]
            order_type  = open_positions[i].type
            symbol = open_positions[i].symbol
            volume = open_positions[i].volume

            if(order_type == mt5.ORDER_TYPE_BUY):
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            
            close_request={
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": order_type,
                "position": deal_id,
                "price": price,
                #"magic": 240000,
                "comment": "Close trade",
                "type_time": mt5.ORDER_TIME_GTC,
                #"type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(close_request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return print("Failed to close order :(")
            else:
                return print (f"Order for {symbol} successfully closed!")

# All close positions
def closeall_position(positions,threshold_price=30):
    # Get the current equity
    equity = mt5.account_info().equity

    # Get the current balance
    balance = mt5.account_info().balance

    # Calculate the equity difference
    equity_difference = equity - balance

    # Check if equity is greater than or equal to the threshold amount
    if equity_difference >= threshold_price:
        # Get all open positions
        #closeall_position(positions)

        for i in range (len(positions)):
            position=positions[i]
            tick = mt5.symbol_info_tick(position.symbol)

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": position.ticket,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL,
                "price": tick.ask if position.type == 1 else tick.bid,  
                "deviation": 20,
                "magic": 240000,
                "comment": "python script close",
                "type_time": mt5.ORDER_TIME_GTC,
                #"type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
        return result

def market_types(Synth_list):    
    lis = []
    for i in range(len(Synth_list)):
        j = 0

        lend = ""
        while Synth_list[i].path[j] != "\\":
            
            lend = f'{lend}'+Synth_list[i].path[j]
            j += 1
        lis.append(lend)

    condensed = list(set(lis))

    secondary = []
    for j in range(len(condensed)):
        initial = []
        for i in range(len(lis)):
            if lis[i] == condensed[j]:
                initial.append(Synth_list[i].name)
        secondary.append(initial)
    
    return condensed, secondary

# def detect_zscore_exit_signals(df, z_col='Z', level=100, signal_col='Final_signal'):
#     """
#     Adds a column `z_exit_signal` to df, indicating when a z-score-based exit should occur.
#     +1 means exit buy, -1 means exit sell.
#     """
#     df = df.copy()
#     df['z_exit_signal'] = 0
#     in_positive_zone = False
#     in_negative_zone = False
#     peak_z = None
#     dip_z = None

#     for i in range(1, len(df)):
#         z = df[z_col].iloc[i]

#         # For Z > 2 (exit buy logic)
#         if z > level:
#             if not in_positive_zone:
#                 in_positive_zone = True
#                 peak_z = z
#             else:
#                 peak_z = max(peak_z, z)
#             # Check for drop halfway from peak
#             if z < level + (peak_z - level) / 2:
#                 # Confirm there's a buy to exit
#                 past_signals = df[signal_col].iloc[:i]
#                 if 2 in past_signals.values:
#                     df.at[df.index[i], 'z_exit_signal'] = 1
#                 in_positive_zone = False
#                 peak_z = None

#         elif in_positive_zone:
#             in_positive_zone = False
#             peak_z = None

#         # For Z < -2 (exit sell logic)
#         if z < -level:
#             if not in_negative_zone:
#                 in_negative_zone = True
#                 dip_z = z
#             else:
#                 dip_z = min(dip_z, z)
#             # Check for rise halfway from dip
#             if z > -level + (dip_z + level) / 2:
#                 # Confirm there's a sell to exit
#                 past_signals = df[signal_col].iloc[:i]
#                 if 1 in past_signals.values:
#                     df.at[df.index[i], 'z_exit_signal'] = -1
#                 in_negative_zone = False
#                 dip_z = None

#         elif in_negative_zone:
#             in_negative_zone = False
#             dip_z = None

#     return df

def detect_zscore_exit_signals(df, z_col='Z', confirm_col='Confirm', level=100, signal_col='Final_signal'):
    """
    Adds a column `z_exit_signal` to df, indicating when a z-score-based exit should occur.
    +1 means exit buy, -1 means exit sell.

    Exit logic is:
    - For a Buy (Final_signal == 2), exit when:
        Z < +100 AND Confirm < 0
    - For a Sell (Final_signal == 1), exit when:
        Z > -100 AND Confirm > 0
    """
    df = df.copy()
    df['z_exit_signal'] = 0
    in_positive_zone = False
    in_negative_zone = False

    for i in range(1, len(df)):
        z = df[z_col].iloc[i]
        confirm = df[confirm_col].iloc[i]

        # Check for potential Buy Exit
        if z > level:
            in_positive_zone = True

        elif in_positive_zone and z < level and confirm < 0:
            past_signals = df[signal_col].iloc[:i]
            if 2 in past_signals.values:  # Buy occurred before
                df.at[df.index[i], 'z_exit_signal'] = 1
            in_positive_zone = False

        # Check for potential Sell Exit
        if z < -level:
            in_negative_zone = True

        elif in_negative_zone and z > -level and confirm > 0:
            past_signals = df[signal_col].iloc[:i]
            if 1 in past_signals.values:  # Sell occurred before
                df.at[df.index[i], 'z_exit_signal'] = -1
            in_negative_zone = False

    return df

def calculate_z_score(series, window=34):
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    z_score = (series - mean)/std
    return z_score

def crossover_lines(df, short_mavg, long_mavg):
    # Generate buy and sell signals based on the MA crossover   
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0
    signals['short_mavg'] = short_mavg
    signals['long_mavg'] = long_mavg
    signals['signal'] = np.where(signals['short_mavg'] > signals['long_mavg'], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    
    return signals

def swing_df(df, slope_line):    
    # Identify ranges for downward and upward slopes
    ranges = []
    current_range = []
    current_type = None

    data = pd.DataFrame()
    data['z_wave_slope']=slope_line.reindex(df.index).interpolate()#k_wave.diff().reindex(df.index).interpolate()

    zero_line = 0
    df_ = df[-len(data):].copy()

    for i, slope in enumerate(data['z_wave_slope']):
        if slope < 0:  # Downward slope
            if current_type != 'down':
                if current_range:
                    ranges.append((current_type, current_range))
                current_range = []
                current_type = 'down'
            current_range.append(i)
        elif slope > 0:  # Upward slope
            if current_type != 'up':
                if current_range:
                    ranges.append((current_type, current_range))
                current_range = []
                current_type = 'up'
            current_range.append(i)

    # Append the last range
    if current_range:
        ranges.append((current_type, current_range))

    # Analyze swings and classify above/below zero
    results = []
    for swing_type, indices in ranges:
        swing_slopes = data.iloc[indices]['z_wave_slope']
        
        swing_prices = df_.iloc[indices]['Close']
        swing_distance = abs(swing_prices.max() - swing_prices.min())
        
        # swing_high = df_.iloc[indices]['High']
        # swing_low = df_.iloc[indices]['Low']
        # swing_distance = abs(swing_high.max() - swing_low.min())
        
        swing_dates = df_.index[indices]
        
        # Determine majority above or below zero
        above_count = sum(1 for slope in swing_slopes if slope > zero_line)
        below_count = sum(1 for slope in swing_slopes if slope <= zero_line)
        swing_location = 'above' if above_count > below_count else 'below'
        
        results.append({
            'Swing Type': swing_type,
            'Start Date': swing_dates[0],
            'End Date': swing_dates[-1],
            'Swing Distance': swing_distance,
            'Highest Close': swing_prices.max(),
            'Lowest Close': swing_prices.min(),
            'Swing Location': swing_location,
            'Indices': indices
        })

    # Create a DataFrame from the results
    swings_df = pd.DataFrame(results)

    return df_, swings_df, results

def the_swinger(df):
    # Parameters
    zero_line = 0
    threshold = 0.1  # Threshold for swing detection

    slopes = df['super_slope']

    # Identify Swing Indices
    up_swings = []
    down_swings = []

    current_swing_type = "up" if slopes.iloc[0] > 0 else "down"
    start_idx = 0

    for i in range(1, len(slopes)):
        if (current_swing_type == "up" and slopes.iloc[i] < -threshold) or (current_swing_type == "down" and slopes.iloc[i] > threshold):
            end_idx = i - 1
            if current_swing_type == "up":
                up_swings.append((start_idx, end_idx))
            else:
                down_swings.append((start_idx, end_idx))
            current_swing_type = "down" if current_swing_type == "up" else "up"
            start_idx = i

    # Add last swing
    end_idx = len(slopes) - 1
    if current_swing_type == "up":
        up_swings.append((start_idx, end_idx))
    else:
        down_swings.append((start_idx, end_idx))

    # Analyze Swings with MACD
    results = []

    for swing_type, swings in [("up", up_swings), ("down", down_swings)]:
        for start_idx, end_idx in swings:
            indices = range(start_idx, end_idx + 1)
            macd_values = df.iloc[indices]["super"]
            majority_position = "above" if (macd_values > zero_line).sum() > (macd_values <= zero_line).sum() else "below"
            results.append({
                "Swing Type": swing_type,
                "Start Index": start_idx,
                "End Index": end_idx,
                "MACD Majority": majority_position,
                "MACD Values": macd_values.tolist()
            })
            
    # Create DataFrame
    swing_analysis_df = pd.DataFrame(results)

    # Calculate Swing Distances
    swing_distances = []
    for swing in results:
        start_idx = swing["Start Index"]
        end_idx = swing["End Index"]
        swing_type = swing["Swing Type"]
        majority_position = swing["MACD Majority"]
        
        start_price = df.iloc[start_idx]["Close"]
        end_price = df.iloc[end_idx]["Close"]
        
        distance = abs(end_price - start_price)
        
        swing_distances.append({
            "Swing Type": swing_type,
            "MACD Majority": majority_position,
            "Distance": distance
        })

    # Create DataFrame for Swing Distances
    swing_distance_df = pd.DataFrame(swing_distances)

    # Calculate Averages
    average_distances = (
        swing_distance_df.groupby(["MACD Majority", "Swing Type"])["Distance"]
        .mean()
        .reset_index()
        .rename(columns={"Distance": "Average Distance"})
    )
    
    return results, swing_analysis_df, swing_distance_df, average_distances

def peak_vall(df_02):
    # Find peaks and valleys
    peaks, _ = find_peaks(df_02.close)
    peak_price = pd.DataFrame(df_02['high'].iloc[peaks])
    valleys, _ = find_peaks(-df_02.close)
    valley_price = pd.DataFrame(df_02['low'].iloc[valleys])

    highs_lows = pd.DataFrame(index=df_02.index)
    highs_lows['peaks'] = peak_price
    highs_lows['valleys'] = valley_price

    highs_lows.fillna(0, inplace=True)

    sig = []
    for i in range(len(highs_lows)):
        if highs_lows['peaks'].iloc[i] != 0:
            sig.append(1)
            
        elif highs_lows['valleys'].iloc[i] != 0:
            sig.append(2)
        
        else:
            sig.append(0)
            
    highs_lows['Signal'] = sig

    return highs_lows

def peak_vall_02(df_02):
    # Find peaks and valleys
    peaks, _ = find_peaks(df_02)
    peak_price = pd.DataFrame(df_02.iloc[peaks])
    valleys, _ = find_peaks(-df_02)
    valley_price = pd.DataFrame(df_02.iloc[valleys])

    highs_lows = pd.DataFrame(index=df_02.index)
    highs_lows['peaks'] = peak_price
    highs_lows['valleys'] = valley_price

    highs_lows.fillna(0, inplace=True)

    sig = []
    for i in range(len(highs_lows)):
        if highs_lows['peaks'].iloc[i] != 0:
            sig.append(1)
            
        elif highs_lows['valleys'].iloc[i] != 0:
            sig.append(2)
        
        else:
            sig.append(0)
            
    highs_lows['Signal'] = sig

    return highs_lows

def stop_levels(Symbol, Synth_list):

    for i in range(len(Synth_list)):
        if Symbol == Synth_list[i].name:

            stop_level = Synth_list[i].trade_stops_level
            lot_size_min = Synth_list[i].volume_min
            lot_size_max =Synth_list[i].volume_max
    
    return stop_level, lot_size_min, lot_size_max

def simple_signal(df, current_candle):
    current_pos = current_candle#df.index.get_loc(current_candle)

    c0 = df['Open'].iloc[current_pos] > df['Close'].iloc[current_pos]
    # Condition 1: The high is greater than the high of the previous day
    c1 = df['High'].iloc[current_pos] > df['High'].iloc[current_pos - 1]
    # Condition 2: The low is less than the low of the previous day
    c2 = df['Low'].iloc[current_pos] < df['Low'].iloc[current_pos - 1]
    # Condition 3: The close of the Outside Bar is less than the low of the previous day
    c3 = df['Close'].iloc[current_pos] < df['Low'].iloc[current_pos - 1]

    if c0 and c1 and c2 and c3:
        return 2  # Signal for entering a Long trade at the open of the next bar
    
    c0 = df['Open'].iloc[current_pos] < df['Close'].iloc[current_pos]
    # Condition 1: The high is greater than the high of the previous day
    c1 = df['Low'].iloc[current_pos] < df['Low'].iloc[current_pos - 1]
    # Condition 2: The low is less than the low of the previous day
    c2 = df['High'].iloc[current_pos] > df['High'].iloc[current_pos - 1]
    # Condition 3: The close of the Outside Bar is less than the low of the previous day
    c3 = df['Close'].iloc[current_pos] > df['High'].iloc[current_pos - 1]
    
    if c0 and c1 and c2 and c3:
        return 1

def cross_signal(df, current_candle):
    current_pos = current_candle#df.index.get_loc(current_candle)

    if df["crause"].iloc[current_pos] == 1:
        return 1  # Signal for entering a Long trade at the open of the next bar
    
    elif df["crause"].iloc[current_pos] == -1:
        return 2

    return 0

def p_v_signal(df, current_candle):
    current_pos = current_candle
    
    if df['Highs_lows'].iloc[current_pos] == 1:
        return 1

    elif df['Highs_lows'].iloc[current_pos] == 2:
        return 2
    
    return 0

def ICT_signal(df, current_candle):
    current_pos = current_candle#df.index.get_loc(current_candle)
    
    # Condition 1: The Open is lesser than the Close for the previous candles
    c0 = df['Open'].iloc[current_pos-1] < df['Close'].iloc[current_pos-1]

    # Condition 2: The Open is greater than the Close for the last two candles
    c1 = df['Open'].iloc[current_pos-2] > df['Close'].iloc[current_pos-2]
    c2 = df['Open'].iloc[current_pos-3] > df['Close'].iloc[current_pos-3]

    c1a = df['High'].iloc[current_pos-2] > df['Low'].iloc[current_pos-3] and df['High'].iloc[current_pos-2] < df['High'].iloc[current_pos-3]
    c1b = df['Low'].iloc[current_pos-2] < df['Low'].iloc[current_pos-3]
   
    # Condition 3: The low is less than the low of the previous day
    c3 = df['Low'].iloc[current_pos-2] < df['Low'].iloc[current_pos - 1]

    if c0 and c1 and c2 and c3 and c1a and c1b:
        return 2  # Signal for entering a Long trade at the open of the next bar
    
    # Condition 1: The Open is lesser than the Close for the previous candles
    c0 = df['Open'].iloc[current_pos-1] > df['Close'].iloc[current_pos-1]

    # Condition 2: The Open is greater than the Close for the last two candles
    c1 = df['Open'].iloc[current_pos-2] < df['Close'].iloc[current_pos-2]
    c2 = df['Open'].iloc[current_pos-3] < df['Close'].iloc[current_pos-3]

    c1a = df['Low'].iloc[current_pos-2] < df['High'].iloc[current_pos-3] and df['Low'].iloc[current_pos-2] > df['Low'].iloc[current_pos-3]
    c1b = df['High'].iloc[current_pos-2] > df['High'].iloc[current_pos-3]
    
    # Condition 3: The low is less than the low of the previous day
    c3 = df['High'].iloc[current_pos-2] > df['High'].iloc[current_pos - 1]
    
    if c0 and c1 and c2 and c3 and c1a and c1b:
        return 1

    return 0


def Mharris_signal(df, current_candle):
    current_pos = current_candle#df.index.get_loc(current_candle)
    
    # Buy condition
    c1 = df['High'].iloc[current_pos] > df['High'].iloc[current_pos-1]
    c2 = df['High'].iloc[current_pos-1] > df['Low'].iloc[current_pos]
    c3 = df['Low'].iloc[current_pos] > df['High'].iloc[current_pos-2]
    c4 = df['High'].iloc[current_pos-2] > df['Low'].iloc[current_pos-1]
    c5 = df['Low'].iloc[current_pos-1] > df['High'].iloc[current_pos-3]
    c6 = df['High'].iloc[current_pos-3] > df['Low'].iloc[current_pos-2]
    c7 = df['Low'].iloc[current_pos-2] > df['Low'].iloc[current_pos-3]

    if c1 and c2 and c3 and c4 and c5 and c6 and c7:
        return 2

    # Symmetrical conditions for short (sell condition)
    c1 = df['Low'].iloc[current_pos] < df['Low'].iloc[current_pos-1]
    c2 = df['Low'].iloc[current_pos-1] < df['High'].iloc[current_pos]
    c3 = df['High'].iloc[current_pos] < df['Low'].iloc[current_pos-2]
    c4 = df['Low'].iloc[current_pos-2] < df['High'].iloc[current_pos-1]
    c5 = df['High'].iloc[current_pos-1] < df['Low'].iloc[current_pos-3]
    c6 = df['Low'].iloc[current_pos-3] < df['High'].iloc[current_pos-2]
    c7 = df['High'].iloc[current_pos-2] < df['High'].iloc[current_pos-3]

    if c1 and c2 and c3 and c4 and c5 and c6 and c7:
        return 1

    return 0

def stoch_signal(df, current_candle):

    c0 = df['K'].iloc[current_candle] < 20
    c1 = df['K'].iloc[current_candle] > df['D'].iloc[current_candle]

    if c0 and c1:
        return 2        #If signal is Long then buy

    c0 = df['K'].iloc[current_candle] > 80
    c1 = df['K'].iloc[current_candle] < df['D'].iloc[current_candle]

    if c0 and c1:
        return 1        #If signal is Long then buy
    
    return 0

def hori_stoch_signal(df, current_candle):

    c0 = df['K'].iloc[current_candle] < 20
    c1 = df['K'].iloc[current_candle-1] > 20

    if c0 and c1:
        return 2        #If signal is Long then buy

    c0 = df['K'].iloc[current_candle] > 80
    c1 = df['K'].iloc[current_candle-1] < 80
    
    if c0 and c1:
        return 1        #If signal is Long then buy
    
    return 0

def CCI_signal(df, current_candle):
    
    # df['Z'].dropna()

    c0 = df['Z'].iloc[current_candle] < -100
    c1 = df['Z'].iloc[current_candle-1] > -100

    if c0 and c1:
        return 2        #If signal is Long then buy

    c0 = df['Z'].iloc[current_candle] > 100
    c1 = df['Z'].iloc[current_candle-1] < 100
    
    if c0 and c1:
        return 1        #If signal is Long then buy
    
    return 0

def create_signals(df, algo):
    lis = []
    for i in range(len(df)):
        lis.append(algo(df, current_candle=i))

    return lis 

# --- Function to estimate peak detection parameters ---
def estimate_peak_parameters(price_series):
    # Estimate prominence based on standard deviation
    prominence = np.std(price_series) * 0.5  # Adjust the multiplier as needed

    # Estimate minimum distance based on average distance between local maxima
    local_maxima_indices = find_peaks(price_series)[0]
    if len(local_maxima_indices) > 1:
        distances = np.diff(local_maxima_indices)
        min_distance = int(np.mean(distances)) if len(distances) > 0 else 15  # Default value if no peaks found
    else:
        min_distance = 15  # Default value

    return prominence, min_distance

# --- Function to check motive wave rules ---
def check_motive_wave_rules(p0, p1, p2, p3, p4, p5):
    """
    Checks rules for a 5-wave impulse pattern based on provided descriptions.
    p0-p5 are price values at the respective points.
    """
    # Determine primary trend direction from Wave 1
    is_uptrend = p1 > p0

    # Wave 1: Initial move (p0 to p1)
    if is_uptrend:
        if not (p1 > p0): return False
    else: # Downtrend
        if not (p1 < p0): return False

    # Wave 2: Pullback (p1 to p2)
    if is_uptrend:
        if not (p2 < p1 and p2 > p0): return False
    else: # Downtrend
        if not (p2 > p1 and p2 < p0): return False

    # Wave 3: Powerhouse (p2 to p3)
    len_w1 = abs(p1 - p0)
    len_w3 = abs(p3 - p2)
    if is_uptrend:
        if not (p3 > p2): return False
    else: # Downtrend
        if not (p3 < p2): return False

    # Wave 4: Breather (p3 to p4)
    if is_uptrend:
        if not (p4 < p3 and p4 > p1): return False
    else: # Downtrend
        if not (p4 > p3 and p4 < p1): return False

    # Wave 5: Final Push (p4 to p5)
    if is_uptrend:
        if not (p5 > p4): return False
    else: # Downtrend
        if not (p5 < p4): return False

    # Check Wave 3 length
    len_w5 = abs(p5 - p4)
    if len_w3 < len_w1 and len_w3 < len_w5: return False

    # Final check: p3 must extend beyond p1 in the direction of the trend
    if is_uptrend:
        if not (p3 > p1): return False
    else: # Downtrend
        if not (p3 < p1): return False

    return True

# --- Function to check corrective wave rules ---
def check_corrective_wave_rules(p5_val, pa_val, pb_val, pc_val, motive_is_uptrend):
    """
    Checks rules for a 3-wave corrective pattern (A-B-C) based on provided descriptions.
    p5_val is the price at the end of the preceding impulse Wave 5.
    pa_val, pb_val, pc_val are prices at the end of Waves A, B, C.
    motive_is_uptrend indicates the direction of the preceding 5-wave impulse.
    """
    if motive_is_uptrend:
        if not (pa_val < p5_val): return False
        if not (pb_val > pa_val and pb_val < p5_val): return False
        if not (pc_val < pb_val and pc_val <= pa_val): return False
    else:
        if not (pa_val > p5_val): return False
        if not (pb_val < pa_val and pb_val > p5_val): return False
        if not (pc_val > pb_val and pc_val >= pa_val): return False

    return True

# --- Function to identify potential patterns ---
def identify_potential_patterns(data, turning_points_indices):
    identified_motive_waves = []
    identified_corrective_waves = []
    price_series = data['Close']
    turning_points_values = price_series.iloc[turning_points_indices].values
    turning_points_dates = price_series.index[turning_points_indices]

    if len(turning_points_values) >= 6:
        for i in range(len(turning_points_values) - 5):
            p_prices = turning_points_values[i:i+6] # p0, p1, p2, p3, p4, p5
            p_dates = turning_points_dates[i:i+6]

            if check_motive_wave_rules(*p_prices):
                wave_sequence_data = list(zip(p_dates, p_prices))
                identified_motive_waves.append(wave_sequence_data)

                motive_is_uptrend = p_prices[1] > p_prices[0] # Based on Wave 1 of this motive sequence

                # Try to find a corrective wave
                if i + 5 + 3 <= len(turning_points_values) -1 : # Need 3 more points for A, B, C
                    c_p5_price = p_prices[5] # Price of point 5 (end of motive)
                    c_pa_price = turning_points_values[i+6]
                    c_pb_price = turning_points_values[i+7]
                    c_pc_price = turning_points_values[i+8]

                    c_p5_date = p_dates[5]
                    c_pa_date = turning_points_dates[i+6]
                    c_pb_date = turning_points_dates[i+7]
                    c_pc_date = turning_points_dates[i+8]

                    if check_corrective_wave_rules(c_p5_price, c_pa_price, c_pb_price, c_pc_price, motive_is_uptrend):
                        corrective_sequence_data = [
                            (c_p5_date, c_p5_price), (c_pa_date, c_pa_price),
                            (c_pb_date, c_pb_price), (c_pc_date, c_pc_price)
                        ]
                        identified_corrective_waves.append(corrective_sequence_data)

    return identified_motive_waves, identified_corrective_waves

# --- Function to evaluate patterns ---
def evaluate_patterns(identified_motive_waves, identified_corrective_waves):
    # Simple evaluation based on the number of identified waves
    return len(identified_motive_waves) + len(identified_corrective_waves)

def divergence_norm(df_c):    
    
    MA_34 = indie.rolling_mean(df_c.close, 34)

    stretch = df_c.close - MA_34
    atr_series = indie.atr(df_c.rename(columns={'Close': 'close', 'Low': 'low', 'High': 'high', 'Date': 'date', 'Open' : 'open'})).fillna(0)
    norm_stretch = stretch.values/atr_series[-len(stretch):].values

    norm_stretch_df = pd.DataFrame()
    norm_stretch_df['norm_stretch'] = norm_stretch
    norm_stretch_df['date'] = stretch.index
    norm_stretch_df = norm_stretch_df.set_index('date')

    norm_stretch_df = norm_stretch_df[34:]

    stretch_filtered =  savgol_filter(norm_stretch_df['norm_stretch'], 51, 3)

    peaks, _ = find_peaks(stretch_filtered, distance=15, prominence=0.2)
    valleys, _ = find_peaks(-stretch_filtered, distance=15, prominence=0.2)

    norm_stretch_df['stretch_filtered'] = stretch_filtered

    MA_13 = indie.rolling_mean(df_c.close, 13)

    Points = pd.DataFrame()
    Points['MA_13'] = MA_13

    Points['Valley_on_13'] = MA_13[norm_stretch_df.index[valleys]].reindex(Points.index).fillna(0)
    Points['Peak_on_13'] = MA_13[norm_stretch_df.index[peaks]].reindex(Points.index).fillna(0)

    Points['Stretch'] = norm_stretch_df['stretch_filtered']

    Points['Valley_on_stretch'] = norm_stretch_df['stretch_filtered'][norm_stretch_df.index[valleys]].reindex(Points.index).fillna(0)
    Points['Peak_on_stretch'] = norm_stretch_df['stretch_filtered'][norm_stretch_df.index[peaks]].reindex(Points.index).fillna(0)

    Divergence = []
    Div_date = []

    last_valley_idx = None
    last_peak_idx = None

    for i in range(len(Points)):
        row = Points.iloc[i]

        # ---- VALLEY DIVERGENCE CHECK ----
        if row['Valley_on_13'] != 0:
            if last_valley_idx is not None:
                prev_row = Points.iloc[last_valley_idx]

                price_curr = row['Valley_on_13']
                price_prev = prev_row['Valley_on_13']
                stretch_curr = row['Valley_on_stretch']
                stretch_prev = prev_row['Valley_on_stretch']

                if (price_curr < price_prev and stretch_curr > stretch_prev) or \
                (price_curr > price_prev and stretch_curr < stretch_prev):
                    Divergence.append('Under')
                    Div_date.append(Points.index[i])

            last_valley_idx = i

        # ---- PEAK DIVERGENCE CHECK ----
        elif row['Peak_on_13'] != 0:
            if last_peak_idx is not None:
                prev_row = Points.iloc[last_peak_idx]

                price_curr = row['Peak_on_13']
                price_prev = prev_row['Peak_on_13']
                stretch_curr = row['Peak_on_stretch']
                stretch_prev = prev_row['Peak_on_stretch']

                if (price_curr > price_prev and stretch_curr < stretch_prev) or \
                (price_curr < price_prev and stretch_curr > stretch_prev):
                    Divergence.append('Above')
                    Div_date.append(Points.index[i])
                    
            last_peak_idx = i
        
    Points['Divergence'] = np.nan
    Points['Divergence'] = pd.Series(np.nan, index=Points.index, dtype='object')
    Points.loc[Div_date, 'Divergence'] = Divergence

    return peaks, valleys, stretch_filtered, Divergence, Div_date, MA_13, MA_34, norm_stretch_df, Points

def divergence_norm_realtime(df_c):    
    # Calculate Moving Averages
    MA_34 = indie.rolling_mean(df_c.close, 34)
    MA_13 = indie.rolling_mean(df_c.close, 13)

    # Normalize stretch vs ATR
    stretch = df_c.close - MA_34
    atr_series = indie.atr(df_c.rename(columns={
        'Close': 'close', 'Low': 'low', 'High': 'high', 
        'Date': 'date', 'Open': 'open'
    })).fillna(0)
    norm_stretch = stretch.values / atr_series[-len(stretch):].values

    # Convert to DataFrame
    norm_stretch_df = pd.DataFrame({
        'norm_stretch': norm_stretch,
        'date': stretch.index
    }).set_index('date')[34:]

    # Replace laggy smoothing with causal EMA
    stretch_filtered = norm_stretch_df['norm_stretch'].ewm(span=10).mean()

    # Detect real-time turning points (momentum shift)
    stretch_diff = np.diff(stretch_filtered)
    stretch_diff = np.insert(stretch_diff, 0, 0)  # Maintain same length

    peaks, valleys = [], []
    for i in range(2, len(stretch_diff)-1):
        if stretch_diff[i-1] > 0 and stretch_diff[i] < 0:
            peaks.append(i)
        elif stretch_diff[i-1] < 0 and stretch_diff[i] > 0:
            valleys.append(i)

    norm_stretch_df['stretch_filtered'] = stretch_filtered

    # Store divergence analysis results
    Points = pd.DataFrame(index=df_c.index)
    Points['MA_13'] = MA_13
    Points['Stretch'] = stretch_filtered

    Points['Valley_on_13'] = MA_13.iloc[valleys].reindex(Points.index).fillna(0)
    Points['Peak_on_13'] = MA_13.iloc[peaks].reindex(Points.index).fillna(0)

    Points['Valley_on_stretch'] = stretch_filtered.iloc[valleys].reindex(Points.index).fillna(0)
    Points['Peak_on_stretch'] = stretch_filtered.iloc[peaks].reindex(Points.index).fillna(0)

    # Divergence detection loop
    Divergence = []
    Div_date = []
    last_valley_idx = None
    last_peak_idx = None

    for i in range(len(Points)):
        row = Points.iloc[i]

        # VALLEY divergence
        if row['Valley_on_13'] != 0:
            if last_valley_idx is not None:
                prev_row = Points.iloc[last_valley_idx]
                if ((row['Valley_on_13'] < prev_row['Valley_on_13'] and 
                     row['Valley_on_stretch'] > prev_row['Valley_on_stretch']) or
                    (row['Valley_on_13'] > prev_row['Valley_on_13'] and 
                     row['Valley_on_stretch'] < prev_row['Valley_on_stretch'])):
                    Divergence.append('Under')
                    Div_date.append(Points.index[i])
            last_valley_idx = i

        # PEAK divergence
        elif row['Peak_on_13'] != 0:
            if last_peak_idx is not None:
                prev_row = Points.iloc[last_peak_idx]
                if ((row['Peak_on_13'] > prev_row['Peak_on_13'] and 
                     row['Peak_on_stretch'] < prev_row['Peak_on_stretch']) or
                    (row['Peak_on_13'] < prev_row['Peak_on_13'] and 
                     row['Peak_on_stretch'] > prev_row['Peak_on_stretch'])):
                    Divergence.append('Above')
                    Div_date.append(Points.index[i])
            last_peak_idx = i

    # Apply divergence to Points
    Points['Divergence'] = pd.Series(np.nan, index=Points.index, dtype='object')
    Points.loc[Div_date, 'Divergence'] = Divergence

    # Return key outputs
    return peaks, valleys, stretch_filtered, Divergence, Div_date, MA_13, MA_34, norm_stretch_df, Points

def zone_define(df_01, indicate='cci', indi_len = 100, upper_level = 100, lower_level = -100, mid_level = 0):    
    
    if indicate == 'rsi':    
        df_01['rsi'] = indie.rsi(df_01.close, indi_len)
        zones_from = extract_zones(df_01, indicator_col='rsi', price_high_col='high', price_low_col='low',
                                        upper=70, lower=30)
    
    elif indicate == 'cci':
        df_01['cci'] = indie.compute_cci(df_01.rename(columns={'close': 'Close', 'low': 'Low', 'high': 'High', 'date': 'Date', 'open' : 'Open'}), n=indi_len)
        zones_from = extract_zones(df_01, indicator_col='cci', upper=upper_level, lower=lower_level)

    zone_df = pd.DataFrame(zones_from)
    # zone_df['end'] = zone_df['end'].fillna(zone_df['start'] + pd.Timedelta(hours=5))

    open_zone = zone_df.set_index('start')#.drop(columns='start')
    open_zone['range'] = open_zone['high'] - open_zone['low']
    # open_zone = open_zone.reindex(df_01.index, method='ffill')
    
    return zone_df, open_zone

def mark_bias(df, cci_col, upper=100, lower=-100):
    # Create a new DataFrame with the same index (preserving time) and needed columns
    result = pd.DataFrame(index=df.index)
    result['cci_col'] = cci_col
    result['bias'] = None
    result['bias_mark'] = 0

    current_bias = None

    for i in range(len(result)):
        cci_val = result['cci_col'].iloc[i]

        if current_bias is None:
            if cci_val >= upper:
                current_bias = 'up'
                result.loc[result.index[i], 'bias'] = 'up'
                result.loc[result.index[i], 'bias_mark'] = 1
            elif cci_val <= lower:
                current_bias = 'down'
                result.loc[result.index[i], 'bias'] = 'down'
                result.loc[result.index[i], 'bias_mark'] = 1

        elif current_bias == 'up':
            if cci_val <= lower:
                current_bias = 'down'
                result.loc[result.index[i], 'bias'] = 'down'
                result.loc[result.index[i], 'bias_mark'] = 1
            else:
                result.loc[result.index[i], 'bias'] = 'up'

        elif current_bias == 'down':
            if cci_val >= upper:
                current_bias = 'up'
                result.loc[result.index[i], 'bias'] = 'up'
                result.loc[result.index[i], 'bias_mark'] = 1
            else:
                result.loc[result.index[i], 'bias'] = 'down'

    return result

def extract_zones(df, indicator_col='rsi', price_high_col='high', price_low_col='low',
                                 upper=70, lower=30):
    """
    Extracts zones based on indicator threshold crossings and returns start, end,
    and price range during each zone.

    Parameters:
        df (pd.DataFrame): DataFrame with datetime index, indicator, and price columns.
        indicator_col (str): Name of the column for indicator values (e.g., 'rsi', 'cci').
        price_high_col (str): Column name for high price (used for zone high).
        price_low_col (str): Column name for low price (used for zone low).
        upper (float): Upper threshold to enter upper zone.
        lower (float): Lower threshold to enter lower zone.

    Returns:
        List of dicts with keys: start, end, type ('upper'/'lower'), high, low.
    """
    zones = []
    in_zone = None
    zone_start_time = None

    for curr_time, row in df.iterrows():
        value = row[indicator_col]

        # ENTER upper zone
        if in_zone is None and value > upper:
            in_zone = 'upper'
            zone_start_time = curr_time

        # ENTER lower zone
        elif in_zone is None and value < lower:
            in_zone = 'lower'
            zone_start_time = curr_time

        # EXIT upper zone
        elif in_zone == 'upper' and value < upper:
            zone_df = df.loc[zone_start_time:curr_time]
            zones.append({
                'start': zone_start_time,
                'end': curr_time,
                'type': 'upper',
                'high': zone_df[price_high_col].max(),
                'low': zone_df[price_low_col].min(),
            })
            in_zone = None
            zone_start_time = None

        # EXIT lower zone
        elif in_zone == 'lower' and value > lower:
            zone_df = df.loc[zone_start_time:curr_time]
            zones.append({
                'start': zone_start_time,
                'end': curr_time,
                'type': 'lower',
                'high': zone_df[price_high_col].max(),
                'low': zone_df[price_low_col].min(),
            })
            in_zone = None
            zone_start_time = None

    return zones

def track_cci_cross_moves_with_time(cci_series, upper=100, lower=-100):
    """
    Tracks consecutive CCI crosses above +100 and below -100, along with timestamps.

    Parameters:
        cci_series (pd.Series): The CCI values (must have datetime index).
        upper (float): Upper threshold for 'up' cross (default: +100).
        lower (float): Lower threshold for 'down' cross (default: -100).

    Returns:
        List of tuples: [(timestamp, 'up' or 'down'), ...]
    """
    moves = []
    prev_val = cci_series.iloc[0]
    prev_time = cci_series.index[0]

    for curr_time, curr_val in cci_series.iloc[1:].items():
        # Cross above upper from below
        if prev_val <= upper and curr_val > upper:
            if len(moves) == 0 or moves[-1][1] != 'up':
                moves.append([curr_time, 'up'])

        # Cross below lower from above
        elif prev_val >= lower and curr_val < lower:
            if len(moves) == 0 or moves[-1][1] != 'down':
                moves.append([curr_time, 'down'])

        prev_val = curr_val
        prev_time = curr_time

    return moves

def double_point(Points):
    
    Points['Point_on_13'] = Points['Valley_on_13'] + Points['Peak_on_13']
    Points['Point_on_stretch'] = Points['Valley_on_stretch'] + Points['Peak_on_stretch']

    Divergence_01 = []
    Div_date_01 = []

    last_point_idx = None

    for i in range(len(Points)):
        row = Points.iloc[i]        
            # ---- POINTS DIVERGENCE CHECK ----
        if row['Point_on_13'] != 0:
            if last_point_idx is not None:
                prev_row = Points.iloc[last_point_idx]

                price_curr = row['Point_on_13']
                price_prev = prev_row['Point_on_13']
                stretch_curr = row['Point_on_stretch']
                stretch_prev = prev_row['Point_on_stretch']

                if (price_curr < price_prev and stretch_curr > stretch_prev) and (Points['Peak_on_13'].iloc[i] != 0):
                    Divergence_01.append('Down')
                    Div_date_01.append(Points.index[i])
                    
                elif (price_curr > price_prev and stretch_curr < stretch_prev) and (Points['Valley_on_13'].iloc[i] != 0):
                    Divergence_01.append('Up')
                    Div_date_01.append(Points.index[i])

            last_point_idx = i
            
    Points['Divergence_01'] = 0
    Points['Divergence_01'] = pd.Series(0, index=Points.index, dtype='object')
    Points.loc[Div_date_01, 'Divergence_01'] = Divergence_01
    
    return Points    

# Functions that transform data
def smoothing_func(signal_, val=99):        # Smoothing functions in a 99 window
    df = signal_.values#[number:]
        # Check for NaN or infinite values in the input signal
    if np.isnan(df).any() or np.isinf(df).any():
        # Handle or remove the NaN or infinite values in the input signal
        df = np.nan_to_num(df)
        
    df_filtered =  savgol_filter(df, val, 3)
    
    return df_filtered

def signify(df, short_mavg, long_mavg):
    # Generate buy and sell signals based on the MA crossover   
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0
    signals['short_mavg'] = short_mavg
    signals['long_mavg'] = long_mavg
    signals['signal'] = np.where(signals['short_mavg'] > signals['long_mavg'], 1.0, 2.0)
    signals['positions'] = signals['signal'].diff()
    
    return signals

def dev_indicator(data_a, data_b, period_s=34, period_l=100):

    data_01 = data_a
    data_02 = data_b

    data_01 = data_01.rename(columns={'close': 'Close', 'low': 'Low', 'high': 'High', 'date': 'Date', 'open' : 'Open'})
    data_02 = data_02.rename(columns={'close': 'Close', 'low': 'Low', 'high': 'High', 'date': 'Date', 'open' : 'Open'})

    MA14a = indie.rolling_mean(data_01.Close, period_s)
    MA50a = indie.rolling_mean(data_01.Close, period_l)

    MA14b = indie.rolling_mean(data_02.Close, period_s)
    MA14b = MA14b.reindex(MA14a.index, method='ffill')
    MA50b = indie.rolling_mean(data_02.Close, period_l)
    MA50b = MA50b.reindex(MA50a.index, method='ffill')
    
    Deviation =[]
    Dev_date =[]
    for i in range(len(MA14a)):
        if MA14a.iloc[i] < MA50a.iloc[i] and MA14b.iloc[i] <= MA50b.iloc[i]:
            Deviation.append(1.0)
            Dev_date.append(MA14a.index[i])
            
        elif MA14a.iloc[i] > MA50a.iloc[i] and MA14b.iloc[i] >= MA50b.iloc[i]:
            Deviation.append(2.0)
            Dev_date.append(MA14a.index[i])
            
            
        elif (MA14a.iloc[i] < MA50a.iloc[i] and MA14b.iloc[i] >= MA50b.iloc[i]) or (MA14a.iloc[i] > MA50a.iloc[i] and MA14b.iloc[i] <= MA50b.iloc[i]):
            Deviation.append(0.0)
            Dev_date.append(MA14a.index[i])

    Dev = pd.DataFrame()
    Dev['date'] = Dev_date
    Dev['value'] = Deviation
    Dev = Dev.set_index('date')
    
    MAs = pd.DataFrame()
    # MAs['date'] = MA14a.index
    MAs['MA14a'] = MA14a
    MAs['MA14b'] = MA14b
    MAs['MA50a'] = MA50a
    MAs['MA50b'] = MA50b
    
    return Dev, MAs

def supertrend(df, df_upper=True, atr_multiplier=3):
    # Calculate the Upper Band(UB) and the Lower Band(LB)
    # Formular: Supertrend =(High+Low)/2 + (Multiplier)∗(ATR)
    
    data = df.copy()
    
    if df_upper == True:
        data = data.rename(columns={'Close': 'close', 'Low': 'low', 'High': 'high', 'Open' : 'open'})
        
    data['ATR'] = indie.atr(data, window=15)
    
    current_average_high_low = (data['high']+data['low'])/2
    # data = pd.DataFrame()
    
    data.dropna(inplace=True)
    data['basicUpperband'] = current_average_high_low + (atr_multiplier * data['ATR'])
    data['basicLowerband'] = current_average_high_low - (atr_multiplier * data['ATR'])
    first_upperBand_value = data['basicUpperband'].iloc[0]
    first_lowerBand_value = data['basicLowerband'].iloc[0]
    upperBand = [first_upperBand_value]
    lowerBand = [first_lowerBand_value]

    for i in range(1, len(data)):
        if data['basicUpperband'].iloc[i] < upperBand[i-1] or data['close'].iloc[i-1] > upperBand[i-1]:
            upperBand.append(data['basicUpperband'].iloc[i])
        else:
            upperBand.append(upperBand[i-1])

        if data['basicLowerband'].iloc[i] > lowerBand[i-1] or data['close'].iloc[i-1] < lowerBand[i-1]:
            lowerBand.append(data['basicLowerband'].iloc[i])
        else:
            lowerBand.append(lowerBand[i-1])

    # supertrend_df = pd.DataFrame(data.index)
    
    data['upperband'] = upperBand
    data['lowerband'] = lowerBand
    data.drop(['basicUpperband', 'basicLowerband',], axis=1, inplace=True)
    
    return data

def generate_signals(df):
    # Intiate a signals list
    signals = [0]

    # Loop through the dataframe
    for i in range(1 , len(df)):
        if df['close'].iloc[i] > df['upperband'].iloc[i]:
            signals.append(2)
        elif df['close'].iloc[i] < df['lowerband'].iloc[i]:
            signals.append(1)
        else:
            signals.append(signals[i-1])

    # Add the signals list as a new column in the dataframe
    df['signals'] = signals
    df['signals'] = df["signals"].shift(1) #Remove look ahead bias
    return df

def create_positions(df):
    # # We need to shut off (np.nan) data points in the upperband where the signal is not 1
    # df['upperband'][df['signals'] == 1] = np.nan
    # # We need to shut off (np.nan) data points in the lowerband where the signal is not -1
    # df['lowerband'][df['signals'] == -1] = np.nan

    # Create a positions list
    buy_positions = [0]
    sell_positions = [0]

    # Loop through the dataframe
    for i in range(1, len(df)):
        # If the current signal is a 1 (Buy) & the it's not equal to the previous signal
        # Then that is a trend reversal, so we BUY at that current market price
        # We take note of the upperband value
        if df['signals'].iloc[i] == 2 and df['signals'].iloc[i] != df['signals'].iloc[i-1]:
            buy_positions.append(df['close'].iloc[i])
            sell_positions.append(0)
        # If the current signal is a -1 (Sell) & the it's not equal to the previous signal
        # Then that is a trend reversal, so we SELL at that current market price
        elif df['signals'].iloc[i] == 1 and df['signals'].iloc[i] != df['signals'].iloc[i-1]:
            sell_positions.append(df['close'].iloc[i])
            buy_positions.append(0)
        else:
            buy_positions.append(0)
            sell_positions.append(0)

    # Add the positions list as a new column in the dataframe
    df['buy_positions'] = buy_positions
    df['sell_positions'] = sell_positions
    return df

def min_max_normalize(data):
    return (data - min(data)) / (max(data) - min(data))

def Wavelet_compute(stock_data):
    # Compute the Continuous Wavelet Transform (CWT) using the Ricker wavelet
    widths = np.arange(1, 15)
    cwt_result = signal.cwt(stock_data.values, signal.ricker, widths)

    # Extract the relevant CWT coefficients for analysis
    cwt_positive = np.where(cwt_result > 0, cwt_result, 0)
    cwt_negative = np.where(cwt_result < 0, cwt_result, 0)

    # Calculate the buy and sell signals from the CWT coefficients
    buy_signal = pd.Series(np.sum(cwt_positive, axis=0), index=stock_data.index)
    sell_signal = pd.Series(-np.sum(cwt_negative, axis=0), index=stock_data.index)

    # Identify buy and sell signal crossovers
    cross_above = (buy_signal >= sell_signal) & (buy_signal.shift(1) < sell_signal.shift(1))
    cross_below = (buy_signal <= sell_signal) & (buy_signal.shift(1) > sell_signal.shift(1))
    
    return cross_above, cross_below

def smooth_frame(main_00):
    line0 = smoothing_func(main_00, val=35)
    line_df = pd.DataFrame()
    line_df['date'] = main_00.index
    line_df['line0'] = line0
    # line_df_dated = line_df.set_index('date')
    
    return line_df

def smooth_frame_lines(df, line='TDI'):
    
    if line == "TDI":
        tdi = indie.tdi(df.close, rsi_lookback=21, rsi_smooth_len=4)
        tdi['date'] = df.date
        tdi = tdi.set_index('date')
        
        line_0 = smooth_frame(tdi['rsi_smooth'])
        line_1 = smooth_frame(tdi['rsi_signal'])#smooth_frame(tdi['rsi_data'])
        mid_line = smooth_frame(tdi['rsi_bb_mid'])
        
        line_0_dated = line_0.set_index('date')
        line_1_dated = line_1.set_index('date')
        mid_line_dated = mid_line.set_index('date')
        
        return line_0, line_0_dated, line_1, line_1_dated, mid_line, mid_line_dated
    
    elif line == 'MACD':
        MACD = pd.DataFrame()
        MACD['main'] = indie.macd(df.close)['macd']
        MACD['signal'] = indie.macd(df.close)['signal']
        MACD['date'] = df.index
        MACD = MACD.set_index('date')
        
        line_0 = smooth_frame(MACD['main'])
        line_1 = smooth_frame(MACD['signal'])
        
        line_0_dated = line_0.set_index('date')
        line_1_dated = line_1.set_index('date')
    
        return line_0, line_0_dated, line_1, line_1_dated
    
def waveler(df, func='macd', price=True):
    
    if price == True:
        mac_main = indie.macd(df.close)[func]
    
    elif price == False:
        mac_main = df.line0

    # Create Wavelet signals for trend direction
    trend_up, trend_down = Wavelet_compute(mac_main)

    trend_d_ind = df.date[[np.where(trend_down.values == True)][0][0]]
    trend_d_loc_main = mac_main[[np.where(trend_down.values == True)][0][0]]

    trend_u_ind = df.date[[np.where(trend_up.values == True)][0][0]]
    trend_u_loc_main = mac_main[[np.where(trend_up.values == True)][0][0]]
    
    return trend_d_ind, trend_d_loc_main, trend_u_ind, trend_u_loc_main

def waveler_01(df):
    
    mac_main = df

    # Create Wavelet signals for trend direction
    trend_up, trend_down = Wavelet_compute_01(mac_main)

    trend_d_ind = df.index[[np.where(trend_down.values == True)][0][0]]
    trend_d_loc_main = mac_main.iloc[[np.where(trend_down.values == True)][0][0]]

    trend_u_ind = df.index[[np.where(trend_up.values == True)][0][0]]
    trend_u_loc_main = mac_main.iloc[[np.where(trend_up.values == True)][0][0]]
    
    return trend_d_ind, trend_d_loc_main, trend_u_ind, trend_u_loc_main

def Wavelet_compute(stock_data):
    # Compute the Continuous Wavelet Transform (CWT) using the Ricker wavelet
    scales = np.arange(1, 15)

    # Perform CWT using PyWavelets with Mexican Hat wavelet ('mexh')
    coeffs, freqs = pywt.cwt(stock_data.values, scales, 'mexh')

    # Extract the relevant CWT coefficients for analysis
    cwt_positive = np.where(coeffs > 0, coeffs, 0)
    cwt_negative = np.where(coeffs < 0, coeffs, 0)

    # Calculate the buy and sell signals from the CWT coefficients
    buy_signal = pd.Series(np.sum(cwt_positive, axis=0), index=stock_data.index)
    sell_signal = pd.Series(-np.sum(cwt_negative, axis=0), index=stock_data.index)

    # Identify buy and sell signal crossovers
    cross_above = (buy_signal >= sell_signal) & (buy_signal.shift(1) < sell_signal.shift(1))
    cross_below = (buy_signal <= sell_signal) & (buy_signal.shift(1) > sell_signal.shift(1))
    
    return cross_above, cross_below

def Wavelet_compute_01(stock_data):
    # Compute the Continuous Wavelet Transform (CWT) using the Ricker wavelet
    # Define the scales (analogous to widths in scipy)
    scales = np.arange(1, 15)

    # Perform CWT using PyWavelets with Mexican Hat wavelet ('mexh')
    coeffs, freqs = pywt.cwt(stock_data.values, scales, 'mexh')

    # Extract the relevant CWT coefficients for analysis
    cwt_positive = np.where(coeffs > 0, coeffs, 0)
    cwt_negative = np.where(coeffs < 0, coeffs, 0)

    # Calculate the buy and sell signals from the CWT coefficients
    buy_signal = pd.Series(np.sum(cwt_positive, axis=0), index=stock_data.index)
    sell_signal = pd.Series(-np.sum(cwt_negative, axis=0), index=stock_data.index)

    # Identify buy and sell signal crossovers
    cross_above = (buy_signal >= sell_signal) & (buy_signal.shift(1) < sell_signal.shift(1))
    cross_below = (buy_signal <= sell_signal) & (buy_signal.shift(1) > sell_signal.shift(1))
    
    return cross_above, cross_below
    
def cycle_input(df_01, line_01):
    Cycle_frame = pd.DataFrame()
    trend_d_ind, trend_d_loc_main, trend_u_ind, trend_u_loc_main = waveler(line_01, price=False)
    Cycle_frame['date'] = df_01.date
    Cycle_frame['down'] = df_01.date.isin(trend_d_ind)
    Cycle_frame['up'] = df_01.date.isin(trend_u_ind)
    Cycle_frame = Cycle_frame.set_index('date')

    np_list = []
    for i in range(len(Cycle_frame)):
        if Cycle_frame['down'].iloc[i] == True:
            np_list.append(2)
        elif Cycle_frame['up'].iloc[i] == True:
            np_list.append(1)
        else:
            np_list.append(np.nan)
            
    Cycle_frame['eitheror'] = np_list
    Cycle_frame = Cycle_frame.ffill()#na(method='ffill', inplace=True)
    # Cycle_frame = Cycle_frame[1:]
    
    return Cycle_frame

def cycle_single(df_01, line_01):
    Cycle_frame = pd.DataFrame()
    trend_d_ind, trend_d_loc_main, trend_u_ind, trend_u_loc_main = waveler(line_01, price=False)
    Cycle_frame['date'] = df_01.date
    Cycle_frame['down'] = df_01.date.isin(trend_d_ind)
    Cycle_frame['up'] = df_01.date.isin(trend_u_ind)
    Cycle_frame = Cycle_frame.set_index('date')
    
    return Cycle_frame

def convergence_plot(combine_1, line_1_dated1, start_ind_1, line_1_dated3, start_ind_3):

    # Calculate RSI and Moving Average (SMA) on the Close price
    RSI = combine_1
    SMA = line_1_dated1.line0[start_ind_1:]

    # Find peaks and valleys in RSI
    rsi_peaks, _ = find_peaks(RSI)
    rsi_valleys, _ = find_peaks(-RSI)

    # Find peaks and valleys in the Moving Average
    sma_peaks, _ = find_peaks(SMA)
    sma_valleys, _ = find_peaks(-SMA)

    # Visualize the RSI and SMA with their peaks
    plt.figure(figsize=(14, 7))

    # Plot the RSI and its peaks
    plt.subplot(2, 1, 1)
    plt.plot(RSI, label='RSI')
    plt.scatter(RSI.index[rsi_peaks], RSI.iloc[rsi_peaks], color='red', label='RSI Peaks')
    plt.scatter(RSI.index[rsi_valleys], RSI.iloc[rsi_valleys], color='blue', label='RSI Valleys')
    # plt.axhline(70, color='gray', linestyle='--')
    # plt.axhline(30, color='gray', linestyle='--')
    plt.title('RSI with Peaks')
    plt.legend()

    # Plot the Moving Average and its peaks
    plt.subplot(2, 1, 2)
    plt.plot(SMA, label='SMA (14-period)')
    plt.plot(line_1_dated3.line0[start_ind_3:], label='Price', color='gray', alpha=0.5)
    plt.scatter(SMA.index[sma_peaks], SMA.iloc[sma_peaks], color='green', label='SMA Peaks')
    plt.scatter(SMA.index[sma_valleys], SMA.iloc[sma_valleys], color='black', label='SMA Peaks')
    plt.title('Moving Average (SMA) with Peaks')
    plt.legend()

    jik =3
    trend_d_ind, trend_d_loc_main, trend_u_ind, trend_u_loc_main = waveler_01(line_1_dated3.line0)
    plt.scatter(trend_d_ind[-jik:], trend_d_loc_main[-jik:], label='Buy Signal', marker='v', color='r')
    plt.scatter(trend_u_ind[-jik:], trend_u_loc_main[-jik:], label='Buy Signal', marker='^', color='b')

    plt.tight_layout()
    plt.show()

def divergence(y_combined, y_flipped_shifted):

    # Detect peaks and valleys in both the original and shifted flipped waves
    peaks, _ = find_peaks(y_combined)
    valleys, _ = find_peaks(-y_combined)

    flipped_peaks, _ = find_peaks(y_flipped_shifted.line0)
    flipped_valleys, _ = find_peaks(-y_flipped_shifted.line0)

    plt.figure(figsize=(19, 7))
    # Plot the original sine wave and shifted flipped sine wave
    plt.plot(y_combined, label='Original Combined Wave', color='black')
    plt.plot(y_flipped_shifted, label='Shifted Flipped Combined Wave', color='gray')

    # Mark the peaks, valleys, and deviations on the original wave
    plt.plot(y_combined.index[peaks], y_combined.iloc[peaks], 'ro', label='Peaks (Original)')  # Red dots for peaks
    plt.plot(y_combined.index[valleys], y_combined.iloc[valleys], 'bo', label='Valleys (Original)')  # Blue dots for valleys

    # Mark the peaks, valleys, and deviations on the flipped wave
    plt.plot(y_flipped_shifted.index[flipped_peaks], y_flipped_shifted.iloc[flipped_peaks], 'r^', label='Peaks (Flipped)')  # Red triangles for peaks (flipped)
    plt.plot(y_flipped_shifted.index[flipped_valleys], y_flipped_shifted.iloc[flipped_valleys], 'bv', label='Valleys (Flipped)')  # Blue triangles for valleys (flipped)

    # Add labels and legend
    plt.xlabel('Time')
    plt.ylabel('Amplitude (y)')
    plt.title('Original and Shifted Flipped Sine Waves with Peak and Valley Deviations')
    plt.legend(loc='best')
    plt.grid(True)

    # Show the plot
    plt.show()
    
def get_ind(data, dated = '2024-10-09 14:05'):    
    # Date to search for
    search_date = pd.to_datetime(dated)

    # Find the index of the specific date
    if search_date in data.index:
        index_position = data.index.get_loc(search_date)
        
    return index_position

def entry_n_result(df, multiplier=0.1):
    Entries_df = pd.DataFrame()
    Entries_df['Direction'] = df['Final_signal'].iloc[np.where(df['Final_signal']!=0)]
    Entries_df['Entry Price'] = df['Close'].iloc[np.where(df['Final_signal']!=0)]
    Entries_df['Exit Price'] = np.nan

    Entries_df = Entries_df.reset_index()

    tp_sl_result = []

    tp_multiplier = multiplier
    sl_multiplier = multiplier

    for _, entry in Entries_df.iterrows():
        entry_time = entry['Date']
        entry_price = entry['Entry Price']
        direction = entry['Direction']

        # Calculate TP and SL based on the direction
        if direction == 2:
            tp = entry_price + (tp_multiplier * (entry_price / (1 + tp_multiplier)))
            sl = entry_price - (sl_multiplier * (entry_price / (1 + sl_multiplier)))
        elif direction == 1:
            tp = entry_price - (tp_multiplier * (entry_price / (1 + tp_multiplier)))
            sl = entry_price + (sl_multiplier * (entry_price / (1 + sl_multiplier)))

        tp_sl_result.append({'Entry Time': entry_time, 'TP': tp, 'SL': sl})
        
    tp_sl_results = pd.DataFrame(tp_sl_result)
    
    return Entries_df, tp_sl_results

# Function to find exit time and result
def determine_exit(row, df_set):
    # Filter prices after the entry time
    filtered_df = df_set[df_set['Date'] >= row['Entry Time']]
    
    # Check if TP or SL is hit
    for _, price_row in filtered_df.iterrows():
        if price_row['Close'] >= row['TP']:
            return {'Exit Time': price_row['Date'], 'Result': 'TP'}
        elif price_row['Close'] <= row['SL']:
            return {'Exit Time': price_row['Date'], 'Result': 'SL'}
    
    # Default case if no TP or SL is hit
    return {'Exit Time': None, 'Result': 'None'}

def merge_dataframes(df_set, Entries_df, tp_sl_results):
    #
    df_set = df_set.reset_index()
        
    # Apply the function to each row in tp_sl_results
    resultsa = tp_sl_results.apply(lambda row: determine_exit(row, df_set), axis=1)

    # Extract Exit Time and Result into separate columns
    tp_sl_results['Exit Time'] = resultsa.apply(lambda x: x['Exit Time'])
    tp_sl_results['Result'] = resultsa.apply(lambda x: x['Result'])
    tp_sl_results['Entry Price'] = tp_sl_results['Entry Time'].apply(lambda x: df_set.loc[df_set['Date'] == x, 'Close'].values[0])

    tp_sl_results['buy_or_sell'] = Entries_df['Direction']
    tp_sl = tp_sl_results.rename(columns={'Entry Time':'Date'})

    merged_df_list = pd.merge(df_set, tp_sl, on='Date', how='outer').sort_values(by='Date')
    merged_df_list["Result"] = merged_df_list["Result"].replace({"TP": 1, "SL": 2, "None": 2})
    merged_df_list = merged_df_list.drop(columns=['Open', 'High', 'Low', 'volume', 'spread', 'Exit Time', 'buy_or_sell'])
    merged_df_list = merged_df_list.fillna(0)

    return merged_df_list

def ML_models(merged_df):

    # Feature selection (modify based on available indicators)
    features = ['Close', 'ATR', 'z_wave_slope', 'HTF_cyc', 'MTF_cyc', 'z_wave', 'k_wave', 'Final_signal', 'TP', 'SL', 'Entry Price']
    target = 'Result'  # 1 = Hit TP, 0 = Hit SL

    # Prepare dataset
    X = merged_df[features]
    y = merged_df[target]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))

    return model