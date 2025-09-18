import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
from numba import jit

import config # Import configuration

def get_symbol_specific_setting(setting_dict, symbol):
    """
    Retrieves a setting specific to a symbol from a dictionary.
    If the setting is not a dictionary, it is returned directly.
    """
    if isinstance(setting_dict, dict):
        return setting_dict.get(symbol, setting_dict.get('default'))
    return setting_dict

def get_pip_size(symbol):
    """Returns the pip size for a given symbol."""
    if "JPY" in symbol:
        return 0.01
    elif "XAU" in symbol or "XAG" in symbol or "XPT" in symbol:
        return 0.01  # Corrected to 0.01 for precious metals
    else:
        return 0.0001

def calculate_atr(high, low, close, period=14):
    """Calculates the Average True Range (ATR)."""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr.iloc[-1]

@jit(nopython=True)
def _check_fvg_nb(high, low, mitigation_threshold=0.5):
    """Numba-optimized core function to find the last valid FVG."""
    for i in range(len(high) - 3, -1, -1):
        c1_high, c3_low = high[i], low[i+2]
        c3_high, c1_low = high[i+2], low[i]

        is_bullish_fvg = c3_low > c1_high
        is_bearish_fvg = c3_high < c1_low
        
        fvg_top = 0.0
        fvg_bottom = 0.0
        fvg_type = 0 # 0: None, 1: Bullish, 2: Bearish

        if is_bullish_fvg:
            fvg_top = c3_low
            fvg_bottom = c1_high
            fvg_type = 1
        elif is_bearish_fvg:
            fvg_top = c1_low
            fvg_bottom = c3_high
            fvg_type = 2
        
        if fvg_type > 0:
            fvg_size = abs(fvg_top - fvg_bottom)
            if fvg_size == 0: continue

            candles_after_high = high[i+3:]
            candles_after_low = low[i+3:]

            if candles_after_low.size == 0:
                mitigation_percent = 0.0
            else:
                mitigation_px = 0.0
                if fvg_type == 1: # Bullish
                    lowest_point_after = np.min(candles_after_low)
                    mitigation_px = max(0, fvg_top - lowest_point_after)
                else: # Bearish
                    highest_point_after = np.max(candles_after_high)
                    mitigation_px = max(0, highest_point_after - fvg_bottom)
                
                mitigation_percent = mitigation_px / fvg_size
            
            if mitigation_percent < mitigation_threshold:
                return fvg_type, fvg_top, fvg_bottom, mitigation_percent

    return 0, 0.0, 0.0, 0.0

@jit(nopython=True)
def _find_swing_highs_lows_nb(high, low):
    """Numba-optimized core function for finding swing points."""
    highs_indices, lows_indices = [], []
    for i in range(1, len(high) - 1):
        if high[i] > high[i-1] and high[i] > high[i+1]:
            highs_indices.append(i)
        if low[i] < low[i-1] and low[i] < low[i+1]:
            lows_indices.append(i)
    return highs_indices, lows_indices

# Wrapper functions

def determine_daily_bias(daily_data):
    """Determines the daily bias (Long/Short/No Trade) based on previous day's close vs PDH/PDL."""
    prev_day = daily_data.iloc[-2]
    pdh, pdl = prev_day['High'], prev_day['Low']
    current_close, current_high, current_low = daily_data.iloc[-1]['Close'], daily_data.iloc[-1]['High'], daily_data.iloc[-1]['Low']

    if current_high >= pdh and current_close < pdh: return 'SHORT_REVERSAL'
    if current_low <= pdl and current_close > pdl: return 'LONG_REVERSAL'
    if current_close >= pdh: return 'LONG_CONTINUATION'
    if current_close <= pdl: return 'SHORT_CONTINUATION'
    if (current_high >= pdh and current_low <= pdl) or (current_high < pdh and current_low > pdl): return 'NO_TRADE_CONSOLIDATION'
    return 'NO_TRADE_OTHER'

def is_in_killer_zone(dt_obj, mode):
    """Checks if a datetime object (localized to NYT) falls within a Killer Zone."""
    if not config.KILLER_ZONE_FILTER[mode]:
        return True  # Skip filter if disabled for the current mode
    if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
        dt_obj = config.NY_TIMEZONE.localize(dt_obj)
    ny_time = dt_obj.astimezone(config.NY_TIMEZONE).time()
    for start_time, end_time in config.KILLER_ZONES:
        if start_time <= ny_time <= end_time: return True
    return False

def check_fvg(candles, mitigation_threshold=0.5):
    """Wrapper for the Numba-optimized FVG check."""
    if len(candles) < 4: return None
    high, low = candles['High'].to_numpy(), candles['Low'].to_numpy()
    fvg_type_code, fvg_top, fvg_bottom, mitigation_percent = _check_fvg_nb(high, low, mitigation_threshold)
    
    if fvg_type_code > 0:
        fvg_type = 'BULLISH' if fvg_type_code == 1 else 'BEARISH'
        return {'type': fvg_type, 'top': fvg_top, 'bottom': fvg_bottom, 'mitigation_percent': mitigation_percent}
    return None

def check_fvg_location(fvg_info, daily_open_price, daily_bias, mode):
    """Checks if FVG is correctly positioned relative to the Daily Open Price."""
    if not config.FVG_LOCATION_RULE[mode]:
        return True  # Skip filter if disabled for the current mode
    if "LONG" in daily_bias: return fvg_info['top'] <= daily_open_price
    if "SHORT" in daily_bias: return fvg_info['bottom'] >= daily_open_price
    return False

def find_swing_highs_lows(candles):
    """Wrapper for the Numba-optimized swing point finder."""
    if len(candles) < 3: return [], []
    high, low = candles['High'].to_numpy(), candles['Low'].to_numpy()
    high_indices, low_indices = _find_swing_highs_lows_nb(high, low)
    
    highs = [(candles.index[i], high[i]) for i in high_indices]
    lows = [(candles.index[i], low[i]) for i in low_indices]
    return highs, lows

def check_ob(candles):
    """Identifies the most recent Order Block (OB) from the provided candles."""
    for i in range(len(candles) - 2, 0, -1):
        c1, c2 = candles.iloc[i], candles.iloc[i+1]
        is_bullish_ob = c1['Close'] < c1['Open'] and c2['Close'] > c2['Open'] and c2['Close'] > c1['High']
        if is_bullish_ob:
            ob_top, ob_bottom = c1['High'], c1['Low']
            if not any(candles['Low'].iloc[i+2:] <= ob_top):
                return {'type': 'BULLISH', 'top': ob_top, 'bottom': ob_bottom}

        is_bearish_ob = c1['Close'] > c1['Open'] and c2['Close'] < c2['Open'] and c2['Close'] < c1['Low']
        if is_bearish_ob:
            ob_top, ob_bottom = c1['High'], c1['Low']
            if not any(candles['High'].iloc[i+2:] >= ob_bottom):
                return {'type': 'BEARISH', 'top': ob_top, 'bottom': ob_bottom}
    return None

def get_trade_decision(symbol, htf_data, ltf_data, daily_bias, daily_open_price, pdh, pdl, mode, tp_rule, sl_buffer_pips, max_sl_pips):
    """Main function to decide on a trade based on the strategy."""
    fvg_info = check_fvg(htf_data)
    if not fvg_info or not check_fvg_location(fvg_info, daily_open_price, daily_bias, mode):
        return None

    pip_size = get_pip_size(symbol)
    sl_buffer = sl_buffer_pips * pip_size

    trade_details = check_mss_and_immediate_entry(ltf_data, daily_bias, sl_buffer, pdh, pdl, tp_rule, symbol, max_sl_pips, pip_size, sl_buffer_pips)
    
    if trade_details:
        risk = abs(trade_details['entry_price'] - trade_details['sl_price'])
        sl_pips = risk / pip_size

        if sl_pips > max_sl_pips:
            print(f"[{datetime.now(pytz.utc)}] Trade on {symbol} invalidated: SL of {sl_pips:.2f} pips exceeds max of {max_sl_pips} pips.")
            return "MAX_SL_EXCEEDED"

    return trade_details

def check_mss_and_immediate_entry(ltf_data, daily_bias, sl_buffer, pdh, pdl, tp_rule, symbol, max_sl_pips, pip_size, sl_buffer_pips):
    """
    Checks for a Market Structure Shift (MSS) and enters immediately on the break.
    TP is customizable, defaulting to 1:3 R:R.
    """
    # Get trade direction restriction for the symbol
    direction_restriction = config.TRADE_DIRECTION_RESTRICTION.get(symbol)

    # Check restriction against daily bias
    if "LONG" in daily_bias and direction_restriction == "sells":
        print(f"[{datetime.now(pytz.utc)}] Trade on {symbol} invalidated: Restricted to sells only.")
        return None
    if "SHORT" in daily_bias and direction_restriction == "buys":
        print(f"[{datetime.now(pytz.utc)}] Trade on {symbol} invalidated: Restricted to buys only.")
        return None

    swing_highs, swing_lows = find_swing_highs_lows(ltf_data)
    
    if ltf_data.empty:
        return None
    
    last_candle = ltf_data.iloc[-1]

    if "LONG" in daily_bias:
        if not swing_highs or not swing_lows:
            return None
        
        sh_price = swing_highs[-1][1]
        
        if last_candle['Close'] > sh_price:
            relevant_lows = [sl for sl in swing_lows if sl[0] < swing_highs[-1][0]]
            if not relevant_lows:
                return None
            sl_def_price = relevant_lows[-1][1]

            entry_price = last_candle['High']
            
            sl_price = sl_def_price - sl_buffer
            risk = abs(entry_price - sl_price)

            if risk == 0:
                return None

            if tp_rule.startswith('1:'):
                try:
                    rr_ratio = float(tp_rule.split(':')[1])
                    tp_price = entry_price + (risk * rr_ratio)
                except (ValueError, IndexError):
                    tp_price = entry_price + (risk * 3) 
            elif tp_rule == 'PDH/PDL':
                tp_price = pdh
            else:
                tp_price = entry_price + (risk * 3)

            print(f"DEBUG (LONG): sl_def_price={sl_def_price}, sl_buffer_pips={sl_buffer_pips}, sl_buffer_price_units={sl_buffer}, entry_price={entry_price}, sl_price={sl_price}, tp_price={tp_price}")

            if tp_price <= entry_price:
                return None

            return {'entry_price': entry_price, 'sl_price': sl_price, 'tp_price': tp_price, 'tp_rule': tp_rule}
        return None

    elif "SHORT" in daily_bias:
        if not swing_lows or not swing_highs:
            return None
            
        sl_price_mss = swing_lows[-1][1]

        if last_candle['Close'] < sl_price_mss:
            relevant_highs = [sh for sh in swing_highs if sh[0] < swing_lows[-1][0]]
            if not relevant_highs:
                return None
            sl_def_price = relevant_highs[-1][1]

            entry_price = last_candle['Low']

            sl_price = sl_def_price + sl_buffer
            risk = abs(entry_price - sl_price)

            if risk == 0:
                return None

            if tp_rule.startswith('1:'):
                try:
                    rr_ratio = float(tp_rule.split(':')[1])
                    tp_price = entry_price - (risk * rr_ratio)
                except (ValueError, IndexError):
                    tp_price = entry_price - (risk * 3)
            elif tp_rule == 'PDH/PDL':
                tp_price = pdl
            else:
                tp_price = entry_price - (risk * 3)

            print(f"DEBUG (SHORT): sl_def_price={sl_def_price}, sl_buffer_pips={sl_buffer_pips}, sl_buffer_price_units={sl_buffer}, entry_price={entry_price}, sl_price={sl_price}, tp_price={tp_price}")

            if tp_price >= entry_price:
                return None

            return {'entry_price': entry_price, 'sl_price': sl_price, 'tp_price': tp_price, 'tp_rule': tp_rule}
    return None