import os
import pandas as pd
import requests
from datetime import datetime
from config import BASE_URL, SYMBOLS, INTERVAL, START_DATE, END_DATE, OHLCV_DIR, TECHNICALS_DIR

# Constants
SECONDS_PER_HOUR = 60 * 60

# Function to fetch historical data with pagination
def fetch_binance_data(symbol, interval, start_date, end_date):
    file_path = f"{OHLCV_DIR}/{symbol}.csv"
    if os.path.exists(file_path):
        print(f"Price data for {symbol} already exists. Loading from file...")
        return pd.read_csv(file_path, parse_dates=['timestamp'])
    
    print(f"Fetching data for {symbol}...")
    start_time = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()) * 1000
    end_time = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()) * 1000
    all_data = []
    
    while True:
        url = f"{BASE_URL}?symbol={symbol}&interval={interval}&startTime={start_time}&limit=1000"  
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

# Fetch and preprocess data
def main():
    all_data = {}
    print("\nFetching historical data...")
    for symbol in SYMBOLS:
        fetch_binance_data(symbol, INTERVAL, START_DATE, END_DATE)
    
    print("Historical data fetched!")
    return all_data

if __name__ == "__main__":
    main()
