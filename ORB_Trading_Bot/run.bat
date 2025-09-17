
@echo off

echo Select a mode to run the ORB Trading Bot:
echo 1. Live Trading
echo 2. Backtest
echo 3. Optimize
echo 4. Walk-Forward Optimization

set /p mode="Enter mode number: "

if "%mode%"=="1" (
    python main.py --mode live
) else if "%mode%"=="2" (
    python backtest.py --symbol SPY500 --days 30 --plot
) else if "%mode%"=="3" (
    python optimizer.py
) else if "%mode%"=="4" (
    python optimizer_walk_forward.py
) else (
    echo Invalid mode selected.
)

pause
