import os
import pandas as pd
import requests
from datetime import datetime
from config import SPOT_BASE_URL, FUTURES_BASE_URL, SYMBOLS, INTERVAL, START_DATE, END_DATE
from config import TIMEFRAMES
from config import OHLCV_DIR

# Constants
SECONDS_PER_HOUR = 60 * 60
SECONDS_PER_TIMEFRAME = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60
}

# Ensure directories exist for each used timeframe (e.g., 5m, 1h)
os.makedirs(OHLCV_DIR, exist_ok=True)
for timeframe in TIMEFRAMES:
    for market in ["spot", "futures"]:
        os.makedirs(os.path.join(OHLCV_DIR, timeframe, market), exist_ok=True)

# Function to fetch historical data with pagination
def fetch_binance_spot_data(symbol, interval, start_date, end_date):
    # File path for saving spot data
    file_path = os.path.join(OHLCV_DIR, interval, "spot", f"{symbol}.csv")
    if os.path.exists(file_path):
        print(f"Spot data for {symbol}, {interval} already exists. Loading from file...")
        return pd.read_csv(file_path, parse_dates=['timestamp'])
    
    print(f"Fetching Spot prices for {symbol}, {interval} timeframe...")
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
        start_time = (int(df['timestamp'].iloc[-1].timestamp()) + SECONDS_PER_TIMEFRAME[interval]) * 1000
        # print(f"Next start time: {pd.to_datetime(start_time, unit='ms')}")
    
    full_df = pd.concat(all_data, ignore_index=True)
    full_df.to_csv(file_path, index=False)
    return full_df

def fetch_binance_futures_data(symbol, interval, start_date, end_date):
    file_path = os.path.join(OHLCV_DIR, interval, "futures", f"{symbol}.csv")
    if os.path.exists(file_path):
        print(f"Futures price data for {symbol}, {interval} already exists. Loading from file...")
        return pd.read_csv(file_path, parse_dates=['timestamp'])

    print(f"Fetching Futures prices for {symbol}, {interval}...")
    start_time = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()) * 1000
    end_time = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()) * 1000
    all_data = []

    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
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
        start_time = int(df['timestamp'].iloc[-1].timestamp() + SECONDS_PER_TIMEFRAME[interval]) * 1000
        print(f"Next start time: {pd.to_datetime(start_time, unit='ms')}")
        # sleep(0.5)  # Avoid hitting API limits

    full_df = pd.concat(all_data, ignore_index=True)
    full_df.to_csv(file_path, index=False)
    return full_df


# Fetch and preprocess data
def main():
    # all_data = {}
    print("\nFetching historical data...")
    for timeframe in TIMEFRAMES:
        for symbol in SYMBOLS:
            fetch_binance_spot_data(symbol, timeframe, START_DATE, END_DATE)
            fetch_binance_futures_data(symbol, timeframe, START_DATE, END_DATE)
        
    print("Historical data fetched!")
    return # all_data

if __name__ == "__main__":
    main()
