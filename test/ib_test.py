from ib_insync import *
import pandas as pd
import os

# Initialize IB object
ib = IB()

def fetch_historical_data(contract, duration, barSize='1 min'):
    """
    Function to fetch historical data for a given duration
    """
    # Request historical data
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=barSize,
        whatToShow='TRADES',  # Consider using 'MIDPOINT' or 'BID_ASK' if needed
        useRTH=True,  # Regular Trading Hours only
        formatDate=1,
        keepUpToDate=False
    )
    
    # Return data or empty list if bars is None
    return bars if bars else []

def save_to_csv(data, filename):
    """
    Save fetched data to a CSV file
    """
    df = util.df(data)
    if df is not None and not df.empty:
        print(df.head())  # Display first few rows of data

        # Create data directory
        if not os.path.exists('data'):
            os.makedirs('data')

        df.to_csv(f'data/{filename}', index=False)
        print(f"Data saved to data/{filename}")
    else:
        print("No data received to save.")

def main():
    # Connect to TWS or IB Gateway
    ib.connect('127.0.0.1', 7496, clientId=1)

    # Define stock contract (ensure exchange is correct)
    contract = Stock('SPY', 'ARCA', 'USD')  # Set exchange to 'ARCA' for SPY

    # Fetch and save data for the past 5 trading days
    bars_5d = fetch_historical_data(contract, duration='5 D')
    save_to_csv(bars_5d, 'SPY_5_days_data.csv')

    # Fetch and save data for the past 7 calendar days
    bars_7d = fetch_historical_data(contract, duration='7 D')
    save_to_csv(bars_7d, 'SPY_7_days_data.csv')

    # Disconnect
    ib.disconnect()

if __name__ == "__main__":
    main()