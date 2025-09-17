# -*- coding: utf-8 -*-
"""
Core trading strategy logic for the ORB Trading Bot.
"""

from datetime import time
import config
import pytz

def get_opening_range(df_day, open_time_str):
    """Identifies the opening range candle for a specific open time and returns its high and low."""
    try:
        market_open = time.fromisoformat(open_time_str)
        orb_candle = df_day.loc[df_day.index.time == market_open].iloc[0]
        orh = orb_candle['High']
        orl = orb_candle['Low']
        print(f"Opening Range for {df_day.index[0].date()} at {open_time_str}: ORH={orh}, ORL={orl}")
        return orh, orl
    except (IndexError, KeyError):
        return None, None

def get_opening_range_24_hour(df_day, current_time):
    """Determines the opening range for 24-hour markets."""
    if config.OPENING_RANGE_METHOD_24_HOUR == 'MIDNIGHT':
        # Use 00:00 UTC as the opening time
        df_day_utc = df_day.tz_convert('UTC')
        return get_opening_range(df_day_utc, '00:00')

    elif config.OPENING_RANGE_METHOD_24_HOUR == 'SESSION':
        # Find the most recent session open time
        most_recent_session_time = None
        for session, open_time_str in config.SESSION_OPEN_TIMES.items():
            session_open_time = time.fromisoformat(open_time_str)
            if current_time.time() >= session_open_time:
                if most_recent_session_time is None or session_open_time > most_recent_session_time:
                    most_recent_session_time = session_open_time
        
        if most_recent_session_time:
            return get_opening_range(df_day, most_recent_session_time.isoformat())
        else:
            return None, None
    else:
        return None, None

def check_trade_signals(df_day, orh, orl, bullish_break, bearish_break):
    """Checks for ORB trade signals (Break and Retest, Reversal)."""
    
    signals = []
    
    if len(df_day) < 2:
        return signals, bullish_break, bearish_break

    prev_candle = df_day.iloc[-2]
    current_candle = df_day.iloc[-1]

    # --- STRATEGY 1: Break and Retest ---
    if not bullish_break and prev_candle['Close'] > orh:
        bullish_break = True
    
    if not bearish_break and prev_candle['Close'] < orl:
        bearish_break = True

    if bullish_break and current_candle['Low'] <= orh and current_candle['Close'] > orh:
        entry_price = orh
        stop_loss = current_candle['Low']
        risk = entry_price - stop_loss
        if risk > 0:
            take_profit = entry_price + (risk * config.RISK_REWARD_RATIO_STANDARD)
            signals.append({'type': 'BUY', 'entry': entry_price, 'sl': stop_loss, 'tp': take_profit, 'strategy': 'Standard'})

    if bearish_break and current_candle['High'] >= orl and current_candle['Close'] < orl:
        entry_price = orl
        stop_loss = current_candle['High']
        risk = stop_loss - entry_price
        if risk > 0:
            take_profit = entry_price - (risk * config.RISK_REWARD_RATIO_STANDARD)
            signals.append({'type': 'SELL', 'entry': entry_price, 'sl': stop_loss, 'tp': take_profit, 'strategy': 'Standard'})

    # --- STRATEGY 2: Reversal (Fakeout) ---
    if config.TRADE_REVERSAL_STRATEGY and not signals:
        if prev_candle['Close'] < orl and current_candle['Close'] > orl:
            entry_price = current_candle['Close']
            stop_loss = current_candle['Low']
            risk = entry_price - stop_loss
            if risk > 0:
                take_profit = entry_price + (risk * config.RISK_REWARD_RATIO_REVERSAL)
                signals.append({'type': 'BUY', 'entry': entry_price, 'sl': stop_loss, 'tp': take_profit, 'strategy': 'Reversal'})

        if prev_candle['Close'] > orh and current_candle['Close'] < orh:
            entry_price = current_candle['Close']
            stop_loss = current_candle['High']
            risk = stop_loss - entry_price
            if risk > 0:
                take_profit = entry_price - (risk * config.RISK_REWARD_RATIO_REVERSAL)
                signals.append({'type': 'SELL', 'entry': entry_price, 'sl': stop_loss, 'tp': take_profit, 'strategy': 'Reversal'})
                
    return signals, bullish_break, bearish_break