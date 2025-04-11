import os
import numpy as np
# import ta
import talib.abstract as ta # use talib.abstract for Freqtrade compatibility
from fetch_data import fetch_binance_spot_data, fetch_binance_futures_data
from config import SYMBOLS, TIMEFRAMES, START_DATE, END_DATE
from config import INDICATORS_DIR
from smc_indicators import add_smc_indicators

# Ensure indicators directories exist for spot and futures
os.makedirs(INDICATORS_DIR, exist_ok=True)
for timeframe in TIMEFRAMES:
    for market in ["spot", "futures"]:
        os.makedirs(os.path.join(INDICATORS_DIR, timeframe, market), exist_ok=True)

# Constants
SECONDS_PER_HOUR = 60 * 60
shift_growth_cols = []

# Function to generate technical indicators
def add_technical_indicators(df):
    # df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)  # ta version
    df['ema50'] = ta.EMA(df['close'], window=50)
    df['ema100'] = ta.EMA(df['close'], window=100)
    df['ema200'] = ta.EMA(df['close'], window=200)
    # df['rsi'] = ta.momentum.rsi(df['close'], window=14)   # ta version
    df['rsi'] = ta.RSI(df['close'], window=14)
    df['ema50_200_bullish'] = (df['ema50'] > df['ema200']).astype(int)
    df['ema50_100_bullish'] = (df['ema50'] > df['ema100']).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['bullish_trend_50'] = (df['close'] > df['ema50']).astype(int)
    df['bullish_trend_100'] = (df['close'] > df['ema100']).astype(int)
    df['bullish_trend_200'] = (df['close'] > df['ema200']).astype(int)
    # shift_growth_cols = []
    for i in range(1, 6):
        df[f'shift_{i}_growth'] = (df['close'].shift(i - 1) > df['close'].shift(i)).astype(int)
        shift_growth_cols.append(f'shift_{i}_growth')
    return df

# Fetch and preprocess data
def prepare_technical_data():
    for timeframe in TIMEFRAMES:
        for symbol in SYMBOLS:
            df_spot = fetch_binance_spot_data(symbol, timeframe, START_DATE, END_DATE)
            df_futures = fetch_binance_futures_data(symbol, timeframe, START_DATE, END_DATE)
            for df in [df_spot, df_futures]:
                df = add_technical_indicators(df)
                df = add_smc_indicators(df)
                df.dropna(inplace=True)
            df_spot.to_csv(os.path.join(INDICATORS_DIR, timeframe, "spot", f"{symbol}.csv"), index=False)
            df_futures.to_csv(os.path.join(INDICATORS_DIR, timeframe, "futures", f"{symbol}.csv"), index=False)
    
    print("Technical indicators added to spot and futures!")
    return

if __name__ == "__main__":
    prepare_technical_data()
