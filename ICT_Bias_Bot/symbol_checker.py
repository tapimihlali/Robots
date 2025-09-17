import MetaTrader5 as mt5
import settings

def run_checker():
    """Connects to MT5 and lists all available symbols."""
    try:
        # Use the same settings as your main bot
        MT5_USER, MT5_PASS, MT5_SERVER, MT5_PATH, _, _ = settings.synthetic()
        if not mt5.initialize(path=MT5_PATH, login=MT5_USER, password=MT5_PASS, server=MT5_SERVER):
            print(f"MT5 initialization failed: {mt5.last_error()}")
            return

        print("Successfully connected to MT5.")
        
        # Get all symbols from the broker
        symbols = mt5.symbols_get()
        if symbols:
            print(f"\n--- Found {len(symbols)} Symbols ---")
            for s in symbols:
                print(s.name)
            print("\n--- End of List ---")
        else:
            print("Could not retrieve symbol list.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure MT5 connection is closed
        mt5.shutdown()
        print("\nConnection closed.")

if __name__ == "__main__":
    run_checker()
