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
    User = 40682668
    Pass = "R0sycheeks@1"
    Server = "Deriv-Demo"
    Path = r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"
    #symbol = ['Step Index','Volatility 10 Index','Volatility 10 (1s) Index', 'Volatility 25 (1s) Index', 'Volatility 25 Index', 'Volatility 75 (1s) Index', 'Volatility 75 Index', 'Volatility 100 Index', 'Volatility 100 (1s) Index', 'Volatility 50 (1s) Index', 'Volatility 90 (1s) Index']
    symbol = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CHF', 'USD/CAD', 'NZD/USD',
 'Wall street 30', 'US SP 500', 'US Tech 100', 'Germany 40', 'UK 100',
 'BTCUSD', 
 'AAPL', 'MSFT', 'TSLA', 'AMZN',
 'XAUUSD', 'XAGUSD', 'XPTUSD',
 'NGAS','UK Brent Oil','US Oil'] 
    timeframe = ['M1', 'M5', 'M15']
    
    return User, Pass, Server, Path, symbol, timeframe

