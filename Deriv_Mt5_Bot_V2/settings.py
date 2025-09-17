# # Forex
def currency():
    User = 71284423
    Pass = "nn4mqrjn"
    Server = 'MetaQuotes-Demo'
    Path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    symbol = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'US100']
    timeframe = ['M1', 'M5', 'M15']

    return User, Pass, Server, Path, symbol, timeframe

# Synthetic
def synthetic():
    User = 40775569
    Pass = "Rosycheeks@1"
    Server = "Deriv-Demo"
    Path = r"C:\Program Files\Deriv\terminal64.exe"
    #symbol = ['Step Index','Volatility 10 Index','Volatility 10 (1s) Index', 'Volatility 25 (1s) Index', 'Volatility 25 Index', 'Volatility 75 (1s) Index', 'Volatility 75 Index', 'Volatility 100 Index', 'Volatility 100 (1s) Index', 'Volatility 50 (1s) Index', 'Volatility 90 (1s) Index']
    symbol = ['Volatility 10 Index','Volatility 25 Index','Volatility 50 Index','Volatility 75 Index','Volatility 100 Index',
 'Volatility 10 (1s) Index','Volatility 15 (1s) Index','Volatility 25 (1s) Index',
 'Volatility 30 (1s) Index','Volatility 50 (1s) Index','Volatility 75 (1s) Index','Volatility 90 (1s) Index','Volatility 100 (1s) Index','Volatility 150 (1s) Index','Volatility 250 (1s) Index',
 'Step Index', 
 'Boom 1000 Index','Boom 500 Index', 'Crash 1000 Index', 'Crash 500 Index',
 'XAUUSD', 'XAGUSD', 'XPTUSD',
 'NGAS','UK Brent Oil','US Oil']
    timeframe = ['M1', 'M5', 'M15']
    
    return User, Pass, Server, Path, symbol, timeframe

