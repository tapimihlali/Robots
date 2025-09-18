import pytz
from datetime import time

# --- TRADING PARAMETERS ---
# Ensure symbol names match your MT5 broker exactly (e.g., 'EURUSD', not 'EUR/USD')
SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD', 'NZDUSD',
    'Wall Street 30', 'US SP 500', 'US Tech 100', 'Germany 40', 'UK 100',
    'BTCUSD', 'AAPL', 'MSFT', 'TSLA', 'AMZN',
    'XAUUSD', 'XAGUSD', 'XPTUSD',
    'NGAS', 'UK Brent Oil', 'USDMXN', 'US Oil', 'US Oil'
]
DAILY_TIMEFRAME = "D1"  # Daily Candlestick for Bias
ENTRY_TIMEFRAMES = ["H4", "H1"] # Higher Timeframes for FVG search

# EMA Periods for plotting and strategy (if applicable)
EMA_ENTRY_PERIOD = 50
EMA_TREND_PERIOD = 200
# Stop Loss and Take Profit settings
SL_BUFFER_PIPS = {
'default': 10,
    'AAPL': 20,
    'AMZN': 20,   
    'AUDUSD': 5,
    'BTCUSD': 10,
    'EURUSD': 5,
    'GBPUSD': 5,
    'Germany 40': 20,
    'MSFT': 10,
    'NGAS': 10,
    'NZDUSD': 5,
    'TSLA': 10,
    'UK 100': 10,
    'US Oil': 20,
    'US SP 500': 20,
    'US Tech 100': 20,
    'USDCAD': 10,
    'USDCHF': 10,
    'USDJPY': 10,
    'USDMXN': 5,
    'Wall Street 30': 20,
    'XAGUSD': 10,
    'XAUUSD': 20,
    'XPTUSD': 20
}

MAX_SL_PIPS = {
    'default': 1000,
    # 'AAPL': 50,
    # 'AMZN': 50,   
    # 'AUDUSD': 10,
    # 'BTCUSD': 50,
    # 'EURUSD': 50,
    # 'GBPUSD': 10,
    # 'Germany 40': 50,
    # 'MSFT': 50,
    # 'NGAS': 50,
    # 'NZDUSD': 10,
    # 'TSLA': 50,
    # 'UK 100': 50,
    # 'US Oil': 50,
    # 'US SP 500': 50,
    # 'US Tech 100': 50,
    # 'USDCAD': 30,
    # 'USDCHF': 20,
    # 'USDJPY': 20,
    # 'USDMXN': 10,
    # 'Wall Street 30': 50,
    # 'XAGUSD': 50,
    # 'XAUUSD': 1000,
    # 'XPTUSD': 50
}

# Dictionary to define custom SL buffer in pips for specific symbols (overrides global)
CUSTOM_SL_BUFFER_PIPS = {
    #"EURUSD": 5,
    #"XAUUSD": 20,
}

# --- TAKE PROFIT RULE ---
# '1:3' for a 1:3 Risk/Reward ratio
# 'PDH/PDL' to target the Previous Day High/Low
TP_RULE = '1:3'

# --- DAILY OPEN TIME (Switch Here) ---
# Use "MIDNIGHT" for 00:00 NYT open
# Use "ASIAN" for 17:00 NYT (start of Asian session) open
DAILY_OPEN_TIME = "ASIAN" 

# --- TIME ZONE AND KILLER ZONES (NYT) ---
NY_TIMEZONE = pytz.timezone('America/New_York')

KILLER_ZONES = [
    (time(2, 0), time(5, 0)),  # London Killer Zone (LKZ)
    (time(7, 0), time(11, 0)), # New York AM Killer Zone (NY AM KZ) - EXTENDED
    (time(13, 0), time(16, 0)) # New York PM Killer Zone (NY PM KZ)
]

# --- LOT SIZING AND RISK ---
RISK_PER_TRADE_PERCENT = 0.01 # 1% Risk if using equity calculation

# Default lot size strategy if not specified in CUSTOM_LOT_SIZES:
# "MIN_LOT": Use the symbol's minimum allowed lot size.
# "RISK_PERCENT": Calculate lot size based on RISK_PER_TRADE_PERCENT.
DEFAULT_LOT_SIZE_STRATEGY = "MIN_LOT"

# Customize lot size per asset (overrides default strategy if specified)
CUSTOM_LOT_SIZES = {
    #"EURUSD": 0.10,
    #"XAUUSD": 0.05,
}

# --- MAX OPEN TRADES ---
MAX_OPEN_TRADES_GLOBAL = 1 # Default max concurrent open trades for any single symbol

# Dictionary to define custom max open trades for specific symbols (overrides global)
CUSTOM_MAX_OPEN_TRADES = {
    #"EURUSD": 2,
}

# --- CUSTOM RISK PARAMETERS (SL Multiplier and TP Rule) ---
# Dictionary to define custom SL multiplier and TP rule for specific symbols.
# Example: 'Symbol': {'sl_multiplier': 2.0, 'tp_rule': '1:2'}
# 'tp_rule' can be '1:1', '1:2', '1:3', '1:4', '1:5', or 'PDH/PDL'
CUSTOM_RISK_PARAMETERS = {
    #"EURUSD": {'sl_multiplier': 2.0, 'tp_rule': '1:3'},
    #"XAUUSD": {'sl_multiplier': 1.0, 'tp_rule': 'PDH/PDL'},
    'AAPL': {'sl_multiplier': 1.0,'tp_rule': 'PDH/PDL'},
    'AMZN': {'sl_multiplier': 1.0,'tp_rule': '1:2'},
    'AUDUSD': {'sl_multiplier': 1.0,'tp_rule': 'PDH/PDL'},
    'BTCUSD': {'sl_multiplier': 1.0,'tp_rule': '1:3'},
    'EURUSD': {'sl_multiplier': 1.0,'tp_rule': 'PDH/PDL'},
    'GBPUSD': {'sl_multiplier': 1.0,'tp_rule': 'PDH/PDL'},
    'Germany 40': {'sl_multiplier': 1.0,'tp_rule': '1:3'},
    'MSFT': {'sl_multiplier': 1.0,'tp_rule': '1:2'},
    'NGAS': {'sl_multiplier': 1.0,'tp_rule': 'PDH/PDL'},
    'NZDUSD': {'sl_multiplier': 1.0,'tp_rule': 'PDH/PDL'},
    'TSLA': {'sl_multiplier': 1.0,'tp_rule': 'PDH/PDL'},
    'UK 100': {'sl_multiplier': 1.0,'tp_rule': 'PDH/PDL'},
    'US Oil': {'sl_multiplier': 1.0,'tp_rule': 'PDH/PDL'},
    'US SP 500': {'sl_multiplier': 1.0,'tp_rule': '1:3'},
    'US Tech 100': {'sl_multiplier': 1.0,'tp_rule': '1:3'},
    'USDCAD': {'sl_multiplier': 1.0,'tp_rule': 'PDH/PDL'},
    'USDCHF': {'sl_multiplier': 1.0,'tp_rule': '1:2'},
    'USDJPY': {'sl_multiplier': 1.0,'tp_rule': 'PDH/PDL'},
    'USDMXN': {'tp_rule': 'PDH/PDL'},
    'Wall Street 30': {'sl_multiplier': 1.0,'tp_rule': '1:3'},
    'XAGUSD': {'sl_multiplier': 1.0,'tp_rule': 'PDH/PDL'},
    'XAUUSD': {'sl_multiplier': 1.0,'tp_rule': '1:3'},
    'XPTUSD': {'sl_multiplier': 1.0,'tp_rule': '1:2'}
}

# --- TRADE DIRECTION RESTRICTION ---
# Dictionary to restrict trading to only buys or sells for specific symbols.
# "buys": Only allow buy trades
# "sells": Only allow sell trades
TRADE_DIRECTION_RESTRICTION = {
    # "EURUSD": "buys",
    # "XAUUSD": "sells",
}

# --- STRATEGY RULES ---
# On/Off switches for FVG location rule for each mode
FVG_LOCATION_RULE = {
    'live': False,
    'backtest': False,
    'optimizer': False,
    'walk_forward': False
}

# On/Off switches for ICT killer timezone filter for each mode
KILLER_ZONE_FILTER = {
    'live': False,
    'backtest': False,
    'optimizer': False,
    'walk_forward': False
}

# --- TRADE MANAGEMENT ---
# --- FVG INVALIDATION EXIT ---
# Dictionary to control the FVG-based invalidation exit logic.
# 'default': True or False - The default setting for all symbols.
# You can override the default for specific symbols by adding them to the dictionary.
# Example: 'EURUSD': False - This would disable invalidation exit for EURUSD,
# while other symbols would use the 'default' setting.
ENABLE_INVALIDATION_EXIT = {
    'default': False,
    # 'EURUSD': True,
    # 'GBPUSD': False,
}

# --- TRAILING STOP LOSS ---
TRAILING_STOP_LOSS = {
    'live_enabled': False, # Enable/disable for live trading
    'backtest_enabled': False, # Enable/disable for backtesting and optimization
    'breakeven_r_multiple': 1.0,  # Move SL to breakeven when profit reaches 1.5R
    'atr_period': 14,
    'atr_multiplier': 2.0
}

# --- BROKER/API SETUP ---
MAGIC_NUMBER = 246810  # Unique identifier for this bot's trades
# Placeholder credentials. Use python-dotenv for production security.
ACCOUNT_ID = "YOUR_ACCOUNT_ID"
PASSWORD = "YOUR_PASSWORD"
SERVER = "YOUR_SERVER"

# --- TELEGRAM NOTIFICATIONS ---
TELEGRAM_ENABLED = True # Set to False to disable notifications
TELEGRAM_BOT_TOKEN = "8358145860:AAE4moHg8RUjk__uj0kZs_rKrCt5wFySWy8" # Replace with your bot token
TELEGRAM_CHAT_ID = "5116546179"       # Replace with your chat ID

# --- OUTPUT SETTINGS ---
OUTPUT_PATH = r"D:\Microsoft VS Code\Projects\2025\ICT_Bias_Outputs" # Base directory for all generated output files