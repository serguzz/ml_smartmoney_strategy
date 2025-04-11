import numpy as np
from scipy.signal import argrelextrema

# Function to add Smart Money Concepts (SMC) indicators
# Add Smart Money indicators BOS and CHOCH for both Long and Short trades
def add_smc_indicators(df):
    STRUCTURE_THRESHOLD = 4  # Threshold for structure distance
    ROLLING_WINDOW = 10  # Rolling window for local extrema detection

    # Find local highs and lows
    local_highs_idx = argrelextrema(df['high'].values, np.greater, order=ROLLING_WINDOW)[0]
    local_lows_idx = argrelextrema(df['low'].values, np.less, order=ROLLING_WINDOW)[0]
    df['local_high'] = np.zeros(len(df))
    df['local_low'] = np.zeros(len(df))
    df.loc[local_highs_idx, 'local_high'] = df.loc[local_highs_idx, 'high']
    df.loc[local_lows_idx, 'local_low'] = df.loc[local_lows_idx, 'low']

    # Forward-fill the last detected high and low to avoid NaNs
    df['prev_high'] = df['local_high'].shift(1).ffill()
    df['prev_low'] = df['local_low'].shift(1).ffill()

    # Track the index of the last local high and low for structure distance calculation
    df['prev_high_idx'] = np.nan
    df['prev_low_idx'] = np.nan
    df.loc[local_highs_idx, 'prev_high_idx'] = local_highs_idx
    df.loc[local_lows_idx, 'prev_low_idx'] = local_lows_idx
    df['prev_high_idx'] = df['prev_high_idx'].shift(1).ffill()
    df['prev_low_idx'] = df['prev_low_idx'].shift(1).ffill()

    # Calculate distance from the previous high and low
    df['prev_high_distance'] = df.index - df['prev_high_idx']
    df['prev_low_distance'] = df.index - df['prev_low_idx']

    # Bullish BOS: Price closes above the previous high in an uptrend
    df['bos_bullish'] = ((df['close'] > df['prev_high']) &
                         (df['prev_high_distance'] > STRUCTURE_THRESHOLD) &
                         (df['bullish_trend_50'] == 1)).astype(int)

    # Bearish BOS: Price closes below the previous low in a downtrend
    df['bos_bearish'] = ((df['close'] < df['prev_low']) &
                      (df['prev_low_distance'] > STRUCTURE_THRESHOLD) &
                      (df['bullish_trend_50'] == 0)).astype(int) # no bullish trend

    # Bullish CHOCH: Price closes above the previous high in a downtrend (trend shift)
    df['choch_bullish'] = ((df['close'] > df['prev_high']) &
                           (df['prev_high_distance'] > STRUCTURE_THRESHOLD) &
                           (df['bullish_trend_50'] == 0)).astype(int)

    # Bearish CHOCH: Price closes below the previous low in an uptrend (trend shift)
    df['choch_bearish'] = ((df['close'] < df['prev_low']) &
                        (df['prev_low_distance'] > STRUCTURE_THRESHOLD) &
                        (df['bullish_trend_50'] == 1)).astype(int) # bullish trend 

    return df
