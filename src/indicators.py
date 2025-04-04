import numpy as np
from scipy.signal import argrelextrema
import ta
from fetch_data import fetch_binance_data
from config import SYMBOLS, INTERVAL, START_DATE, END_DATE, INDICATORS_DIR

# Constants
# SYMBOLS = ["TESTUSDT"]
SECONDS_PER_HOUR = 60 * 60
shift_growth_cols = []

# Function to generate technical indicators
def add_technical_indicators(df):
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema100'] = ta.trend.ema_indicator(df['close'], window=100)
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
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

# Function to generate Smart Money Concepts (BOS and CHOCH)
def add_smc_indicators(df):
    STRUCTURE_THRESHOLD = 4  # Threshold for structure distance
    ROLLING_WINDOW = 10  # Rolling window for local extrema detection

    # Find local highs
    local_highs_idx = argrelextrema(df['high'].values, np.greater, order=ROLLING_WINDOW)[0]
    df['local_high'] = np.nan
    df.loc[local_highs_idx, 'local_high'] = df.loc[local_highs_idx, 'high']

    # Forward-fill the last detected high to avoid NaNs
    df['prev_high'] = df['local_high'].shift(1).ffill()

    # Track the index of the last local high for structure distance calculation
    df['prev_high_idx'] = np.nan
    df.loc[local_highs_idx, 'prev_high_idx'] = local_highs_idx
    df['prev_high_idx'] = df['prev_high_idx'].shift(1).ffill()

    # Calculate distance from the previous high
    df['prev_high_distance'] = df.index - df['prev_high_idx']

    # Bullish BOS: Price closes above the previous high in an uptrend
    df['bos'] = ((df['close'] > df['prev_high']) &
                 (df['prev_high_distance'] > STRUCTURE_THRESHOLD) &
                 (df['bullish_trend_50'] == 1)).astype(int)

    # Bullish CHOCH: Price closes above the previous high in a downtrend (trend shift)
    df['choch'] = ((df['close'] > df['prev_high']) &
                   (df['prev_high_distance'] > STRUCTURE_THRESHOLD) &
                   (df['bullish_trend_50'] == 0)).astype(int)

    return df

# Fetch and preprocess data
def prepare_technical_data():
    all_data = {}
    for symbol in SYMBOLS:
        df = fetch_binance_data(symbol, INTERVAL, START_DATE, END_DATE)
        df = add_technical_indicators(df)
        df = add_smc_indicators(df)
        df.dropna(inplace=True)
        all_data[symbol] = df
        df.to_csv(f"{INDICATORS_DIR}/{symbol}.csv", index=False)
    
    print("Technical indicators added!")
    return all_data

if __name__ == "__main__":
    prepare_technical_data()
