import os
import pandas as pd
import requests
from datetime import datetime
from config import SPOT_BASE_URL, FUTURES_BASE_URL, SYMBOLS, INTERVAL, START_DATE, END_DATE
from config import OHLCV_DIR

# Constants
SECONDS_PER_HOUR = 60 * 60

# Function to fetch historical data with pagination
def fetch_binance_spot_data(symbol, interval, start_date, end_date):
    # File path for saving spot data
    file_path = f"{OHLCV_DIR}/spot/{symbol}.csv"
    if os.path.exists(file_path):
        print(f"Price data for {symbol} already exists. Loading from file...")
        return pd.read_csv(file_path, parse_dates=['timestamp'])
    
    print(f"Fetching data for {symbol}...")
    start_time = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()) * 1000
    end_time = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()) * 1000
    all_data = []
    
    while True:
        url = f"{SPOT_BASE_URL}?symbol={symbol}&interval={interval}&startTime={start_time}&limit=1000"  
        response = requests.get(url)
        data = response.json()
        if start_time >= end_time:
            break
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', '_', '_', '_', '_', '_', '_'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        all_data.append(df)
        start_time = (int(df['timestamp'].iloc[-1].timestamp()) + SECONDS_PER_HOUR) * 1000
    
    full_df = pd.concat(all_data, ignore_index=True)
    full_df.to_csv(file_path, index=False)
    return full_df

def fetch_binance_futures_data(symbol, interval, start_date, end_date):
    file_path = f"{OHLCV_DIR}/futures/{symbol}.csv"
    if os.path.exists(file_path):
        print(f"Futures price data for {symbol} already exists. Loading from file...")
        return pd.read_csv(file_path, parse_dates=['timestamp'])

    print(f"Fetching futures data for {symbol}...")
    start_time = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()) * 1000
    end_time = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()) * 1000
    all_data = []

    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1500
        }
        response = requests.get(FUTURES_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if not data:
            break

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        all_data.append(df)

        # Advance time to next batch (past the last fetched candle)
        start_time = int(df['timestamp'].iloc[-1].timestamp() + SECONDS_PER_HOUR) * 1000
        # sleep(0.5)  # Avoid hitting API limits

    full_df = pd.concat(all_data, ignore_index=True)
    full_df.to_csv(file_path, index=False)
    return full_df


# Fetch and preprocess data
def main():
    # all_data = {}
    print("\nFetching historical data...")
    for symbol in SYMBOLS:
        fetch_binance_spot_data(symbol, INTERVAL, START_DATE, END_DATE)
        fetch_binance_futures_data(symbol, INTERVAL, START_DATE, END_DATE)
    
    print("Historical data fetched!")
    return # all_data

if __name__ == "__main__":
    main()
