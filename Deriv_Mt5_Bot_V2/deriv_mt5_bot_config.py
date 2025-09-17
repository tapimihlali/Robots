import MetaTrader5 as mt5

# -- META TRADER 5 ACCOUNT CREDENTIALS --
# Replace with your actual account details
def get_credentials():
    """
    Returns the MT5 account credentials.
    """
    return {
        "user": 40682668,
        "password": "R0sycheeks@1",
        "server": "Deriv-Demo",
        "path": r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"
    }

# -- TELEGRAM NOTIFICATION CREDENTIALS --
def get_telegram_credentials():
    """
    Returns the Telegram bot token and chat ID.
    """
    return {
        "bot_token": "8233000695:AAH1qiEa4uaD4dJ_dpioK6-RUepfFZ00c4g",
        "chat_id": "5116546179"
    }

# -- SYMBOLS TO TRADE --
def get_symbols():
    """
    Returns the list of symbols to be traded by the bot.
    """
    return [
        'Volatility 10 Index','Volatility 25 Index', 'Volatility 100 Index',
        'Volatility 15 (1s) Index', 'Volatility 25 (1s) Index', 'Volatility 30 (1s) Index', 'Volatility 75 (1s) Index', 'Volatility 90 (1s) Index', 'Volatility 100 (1s) Index', 
        'Boom 1000 Index','Boom 500 Index', 'Crash 1000 Index', 'Crash 500 Index',
        
    ] #'Step Index','Boom 1000 Index','Boom 500 Index', 'Crash 1000 Index', 'Crash 500 Index''Volatility 50 (1s) Index', 'Volatility 10 (1s) Index',

# -- SYMBOLS TO OPTIMIZE --
# List of symbols for the strategy_optimizer.py script.
SYMBOLS_TO_OPTIMIZE = [
        'Volatility 10 Index','Volatility 25 Index', 'Volatility 100 Index',
        'Volatility 10 (1s) Index', 'Volatility 15 (1s) Index', 'Volatility 25 (1s) Index', 'Volatility 30 (1s) Index', 'Volatility 75 (1s) Index', 'Volatility 90 (1s) Index', 'Volatility 100 (1s) Index', 
      
        'Boom 1000 Index', 'Boom 500 Index', 'Crash 1000 Index', 'Crash 500 Index',
] #'NGAS', 'UK Brent Oil', 'US Oil','XPTUSD','Volatility 150 (1s) Index','XAGUSD', 'XPTUSD', 'Volatility 50 Index', 'Volatility 75 Index',  'Volatility 50 (1s) Index','XAUUSD'   'Step Index',

# --- OPTIMIZATION SETTINGS ---
OPTIMIZATION_DAYS = 180  # Number of days of historical data to use for optimization

# --- STRATEGY PARAMETERS ---
TF_TREND = mt5.TIMEFRAME_M5  # Timeframe for determining the main trend (5-minute)
TF_ENTRY = mt5.TIMEFRAME_M1  # Timeframe for finding trade entries (1-minute)
EMA_TREND_PERIOD = 200  # Period for the Exponential Moving Average (EMA) on the trend timeframe
EMA_ENTRY_PERIOD = 50  # Period for the EMA on the entry timeframe

# --- INDICATOR SETTINGS ---
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14
ADX_PERIOD = 14

# --- MAGIC NUMBER ---
MAGIC_NUMBER = 123456  # Unique identifier for trades opened by this bot

# --- RISK MANAGEMENT SETTINGS ---
RISK_PERCENT_PER_TRADE = 1.0  # The percentage of account equity to risk per trade
SL_MULTIPLIER = 2  # Multiplier for the Average True Range (ATR) to set the Stop Loss
TP_RISK_REWARD = 2  # Risk-to-reward ratio for setting the Take Profit
MAX_OPEN_TRADES_GLOBAL = 1 # The default maximum number of concurrent open trades for any single symbol.

# --- STRATEGY ENHANCEMENTS ---
ADX_THRESHOLD = 25  # ADX value above which to consider a trend strong enough to trade
TRAILING_STOP_ATR_MULTIPLIER = 1.5  # Multiplier for ATR to set the trailing stop distance
COMMISSION_PER_LOT = 0.2  # Commission cost per lot (e.g., 0.2 units of the currency)

# --- TRAILING STOP SETTINGS ---
# Enable/disable trailing stop for different modes
LIVE_TRADING_TRAILING_STOP = False
BACKTEST_TRAILING_STOP = False
OPTIMIZATION_TRAILING_STOP = False

# --- CUSTOM SYMBOL SETTINGS ---
# Dictionary to define custom lot sizes for specific symbols
CUSTOM_LOT_SIZES = {
    'Volatility 10 Index': 3.0,
    'Volatility 25 Index': 1.0,
    'Volatility 50 Index': 12.0,
    'Volatility 75 Index': 0.01,
    'Volatility 100 Index': 0.55,
    'Volatility 10 (1s) Index': 1.0,
    'Volatility 15 (1s) Index': 0.4,
    'Volatility 25 (1s) Index': 0.005,
    'Volatility 30 (1s) Index': 0.4,
    'Volatility 75 (1s) Index': 0.3,
    'Volatility 100 (1s) Index': 1.2,
    'Volatility 90 (1s) Index': 0.2,
}

# Dictionary to define custom max open trades for specific symbols
CUSTOM_MAX_OPEN_TRADES = {
    'Volatility 90 (1s) Index': 1,
    'NGAS': 1,
    'XPTUSD': 1,
    'Boom 1000 Index': 1, 
    'Boom 500 Index': 1, 
    'Crash 1000 Index': 1, 
    'Crash 500 Index': 1,
}

# --- TRADE DIRECTION RESTRICTION ---
# Dictionary to restrict trading to only buys or sells for specific symbols.
# "buys": Only allow buy trades
# "sells": Only allow sell trades
TRADE_DIRECTION_RESTRICTION = {
    # "Boom 1000 Index": "buys",
    # "Crash 1000 Index": "sells",
    # "Boom 500 Index": "buys",
    # "Crash 500 Index": "sells",
}

# Dictionary to define custom risk parameters (SL multiplier and TP ratio) for specific symbols 400 days
CUSTOM_RISK_PARAMETERS = {
    # 4th Test TS0 09/09/2025
    'Volatility 10 Index': {'sl_multiplier': 3.0, 'tp_risk_reward': 3.2},
    'Volatility 25 Index': {'sl_multiplier': 2.7, 'tp_risk_reward': 1.5},
    #'Volatility 25 Index': {'sl_multiplier': 2.6, 'tp_risk_reward': 1.1},
    #'Volatility 50 Index': {'sl_multiplier': 2.5, 'tp_risk_reward': 3.5},
    #'Volatility 75 Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 2.0},
    'Volatility 100 Index': {'sl_multiplier': 2.6, 'tp_risk_reward': 1.8},####
    'Volatility 10 (1s) Index': {'sl_multiplier': 2.5, 'tp_risk_reward': 1.7},
    #'Volatility 10 (1s) Index': {'sl_multiplier': 3.0, 'tp_risk_reward': 1.8},
    'Volatility 15 (1s) Index': {'sl_multiplier': 1.4, 'tp_risk_reward': 2.2},
    'Volatility 25 (1s) Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 1.0},###
    'Volatility 30 (1s) Index': {'sl_multiplier': 1.2, 'tp_risk_reward': 1.4},###
    'Volatility 50 (1s) Index': {'sl_multiplier': 2.0, 'tp_risk_reward': 2.0},
    'Volatility 75 (1s) Index': {'sl_multiplier': 2.0, 'tp_risk_reward': 1.4},###
    'Volatility 90 (1s) Index': {'sl_multiplier': 2.0, 'tp_risk_reward': 1.8},###
    'Volatility 100 (1s) Index': {'sl_multiplier': 2.6, 'tp_risk_reward': 2.2},
    #'Volatility 150 (1s) Index': {'sl_multiplier': 1.2, 'tp_risk_reward': 1.2},
    #'Step Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 1.8},
    'Boom 1000 Index': {'sl_multiplier': 2.4, 'tp_risk_reward': 3.1},
    'Boom 500 Index': {'sl_multiplier': 2.6, 'tp_risk_reward': 3.1},
    'Crash 1000 Index': {'sl_multiplier': 2.8, 'tp_risk_reward': 3.0},
    'Crash 500 Index': {'sl_multiplier': 1.4, 'tp_risk_reward': 2.4}
    # 'Boom 1000 Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 4.0},
    # 'Boom 500 Index': {'sl_multiplier': 1.4, 'tp_risk_reward': 4.0},
    # 'Crash 1000 Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 4.0},
    # 'Crash 500 Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 3.2}

}

# --- OPTIMIZED RISK PARAMETERS (Manually update after optimization) ---
# This dictionary will store the best SL_MULTIPLIER and TP_RISK_REWARD for each symbol
# obtained from single-symbol optimization backtests.
OPTIMIZED_RISK_PARAMETERS = {
    # Example: 'Volatility 10 Index': {'sl_multiplier': 2.0, 'tp_risk_reward': 2.0},
    # Example: 'Volatility 25 Index': {'sl_multiplier': 2.7, 'tp_risk_reward': 1.5},
    # Example: 'Volatility 100 Index': {'sl_multiplier': 1.4, 'tp_risk_reward': 3.2},
    
    # 1st Test TS0
    # 'Volatility 10 Index': {'sl_multiplier': 1.7, 'tp_risk_reward': 1.5},
    # 'Volatility 25 Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 1.5},
    # #'Volatility 50 Index': {'sl_multiplier': 2.5, 'tp_risk_reward': 3.5},
    # #'Volatility 75 Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 2.0},
    # 'Volatility 100 Index': {'sl_multiplier': 1.4, 'tp_risk_reward': 3.2},####
    # 'Volatility 10 (1s) Index': {'sl_multiplier': 2.5, 'tp_risk_reward': 3.0},
    # 'Volatility 15 (1s) Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 3.0},
    # 'Volatility 25 (1s) Index': {'sl_multiplier': 1.5, 'tp_risk_reward': 1.0},###
    # 'Volatility 30 (1s) Index': {'sl_multiplier': 2.0, 'tp_risk_reward': 1.0},###
    # #'Volatility 50 (1s) Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 1.5},
    # 'Volatility 75 (1s) Index': {'sl_multiplier': 1.5, 'tp_risk_reward': 1.0},###
    # 'Volatility 90 (1s) Index': {'sl_multiplier': 3.0, 'tp_risk_reward': 2.0},###
    # 'Volatility 100 (1s) Index': {'sl_multiplier': 2.5, 'tp_risk_reward': 3.0},
    # #'Volatility 150 (1s) Index': {'sl_multiplier': 1.2, 'tp_risk_reward': 1.2},
    # #'Step Index': {'sl_multiplier': 1.5, 'tp_risk_reward': 3.5},
    # 'Boom 1000 Index': {'sl_multiplier': 1.5, 'tp_risk_reward': 3.5},
    # 'Boom 500 Index': {'sl_multiplier': 1.5, 'tp_risk_reward': 3},
    # 'Crash 1000 Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 3.5},
    # 'Crash 500 Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 3.5}
    
    # 2nd Test TS0 04/09/2025
    # 'Volatility 10 Index': {'sl_multiplier': 1.7, 'tp_risk_reward': 1.5},
    # 'Volatility 25 Index': {'sl_multiplier': 2.7, 'tp_risk_reward': 1.5},
    # #'Volatility 50 Index': {'sl_multiplier': 2.5, 'tp_risk_reward': 3.5},
    # #'Volatility 75 Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 2.0},
    # 'Volatility 100 Index': {'sl_multiplier': 2.7, 'tp_risk_reward': 1.5},####
    # 'Volatility 10 (1s) Index': {'sl_multiplier': 2.5, 'tp_risk_reward': 1.7},
    # 'Volatility 15 (1s) Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 3.2},
    # 'Volatility 25 (1s) Index': {'sl_multiplier': 1.3, 'tp_risk_reward': 1.0},###
    # 'Volatility 30 (1s) Index': {'sl_multiplier': 2.0, 'tp_risk_reward': 1.1},###
    # #'Volatility 50 (1s) Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 1.5},
    # 'Volatility 75 (1s) Index': {'sl_multiplier': 1.7, 'tp_risk_reward': 1.1},###
    # 'Volatility 90 (1s) Index': {'sl_multiplier': 1.9, 'tp_risk_reward': 1.7},###
    # 'Volatility 100 (1s) Index': {'sl_multiplier': 2.9, 'tp_risk_reward': 1.8},
    # #'Volatility 150 (1s) Index': {'sl_multiplier': 1.2, 'tp_risk_reward': 1.2},
    # #'Step Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 1.8},
    # 'Boom 1000 Index': {'sl_multiplier': 2.4, 'tp_risk_reward': 3.4},
    # 'Boom 500 Index': {'sl_multiplier': 2.6, 'tp_risk_reward': 2.6},
    # 'Crash 1000 Index': {'sl_multiplier': 1.9, 'tp_risk_reward': 3.9},
    # 'Crash 500 Index': {'sl_multiplier': 1.4, 'tp_risk_reward': 2.8}

    # 3rd Test TS0 05/09/2025
    # 'Volatility 10 Index': {'sl_multiplier': 1.7, 'tp_risk_reward': 1.5},
    # 'Volatility 25 Index': {'sl_multiplier': 2.7, 'tp_risk_reward': 1.5},
    # #'Volatility 50 Index': {'sl_multiplier': 2.5, 'tp_risk_reward': 3.5},
    # #'Volatility 75 Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 2.0},
    # 'Volatility 100 Index': {'sl_multiplier': 2.7, 'tp_risk_reward': 2.1},####
    # 'Volatility 10 (1s) Index': {'sl_multiplier': 1.8, 'tp_risk_reward': 2.0},
    # 'Volatility 15 (1s) Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 3.1},
    # 'Volatility 25 (1s) Index': {'sl_multiplier': 2.5, 'tp_risk_reward': 1.5},###
    # 'Volatility 30 (1s) Index': {'sl_multiplier': 2.4, 'tp_risk_reward': 1.1},###
    # #'Volatility 50 (1s) Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 1.5},
    # 'Volatility 75 (1s) Index': {'sl_multiplier': 1.2, 'tp_risk_reward': 1.3},###
    # 'Volatility 90 (1s) Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 1.0},###
    # 'Volatility 100 (1s) Index': {'sl_multiplier': 2.9, 'tp_risk_reward': 1.8},
    # #'Volatility 150 (1s) Index': {'sl_multiplier': 1.2, 'tp_risk_reward': 1.2},
    # #'Step Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 1.8},
    # #'Boom 1000 Index': {'sl_multiplier': 2.4, 'tp_risk_reward': 3.1},
    # #'Boom 500 Index': {'sl_multiplier': 2.6, 'tp_risk_reward': 3.1},
    # #'Crash 1000 Index': {'sl_multiplier': 2.8, 'tp_risk_reward': 3.0},
    # #'Crash 500 Index': {'sl_multiplier': 1.4, 'tp_risk_reward': 2.4}
    
    
    # # 4th Test TS0 09/09/2025
    # 'Volatility 10 Index': {'sl_multiplier': 3.0, 'tp_risk_reward': 3.2},
    # 'Volatility 25 Index': {'sl_multiplier': 2.7, 'tp_risk_reward': 1.5},
    # #'Volatility 25 Index': {'sl_multiplier': 2.6, 'tp_risk_reward': 1.1},
    # #'Volatility 50 Index': {'sl_multiplier': 2.5, 'tp_risk_reward': 3.5},
    # #'Volatility 75 Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 2.0},
    # 'Volatility 100 Index': {'sl_multiplier': 2.6, 'tp_risk_reward': 1.8},####
    # 'Volatility 10 (1s) Index': {'sl_multiplier': 2.5, 'tp_risk_reward': 1.7},
    # #'Volatility 10 (1s) Index': {'sl_multiplier': 3.0, 'tp_risk_reward': 1.8},
    # 'Volatility 15 (1s) Index': {'sl_multiplier': 1.4, 'tp_risk_reward': 2.2},
    # 'Volatility 25 (1s) Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 1.0},###
    # 'Volatility 30 (1s) Index': {'sl_multiplier': 1.2, 'tp_risk_reward': 1.4},###
    # 'Volatility 50 (1s) Index': {'sl_multiplier': 2.0, 'tp_risk_reward': 2.0},
    # 'Volatility 75 (1s) Index': {'sl_multiplier': 2.0, 'tp_risk_reward': 1.4},###
    # 'Volatility 90 (1s) Index': {'sl_multiplier': 2.0, 'tp_risk_reward': 1.8},###
    # 'Volatility 100 (1s) Index': {'sl_multiplier': 2.6, 'tp_risk_reward': 2.2},
    # #'Volatility 150 (1s) Index': {'sl_multiplier': 1.2, 'tp_risk_reward': 1.2},
    # #'Step Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 1.8},
    # 'Boom 1000 Index': {'sl_multiplier': 2.4, 'tp_risk_reward': 3.1},
    # 'Boom 500 Index': {'sl_multiplier': 2.6, 'tp_risk_reward': 3.1},
    # 'Crash 1000 Index': {'sl_multiplier': 2.8, 'tp_risk_reward': 3.0},
    # 'Crash 500 Index': {'sl_multiplier': 1.4, 'tp_risk_reward': 2.4}
    # # 'Boom 1000 Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 4.0},
    # # 'Boom 500 Index': {'sl_multiplier': 1.4, 'tp_risk_reward': 4.0},
    # # 'Crash 1000 Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 4.0},
    # # 'Crash 500 Index': {'sl_multiplier': 1.0, 'tp_risk_reward': 3.2}
}

def get_optimized_risk_parameters():
    """
    Returns the manually updated optimized risk parameters for each symbol.
    """
    return OPTIMIZED_RISK_PARAMETERS