import os

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OHLCV_DIR = os.path.join(BASE_DIR, "../data/ohlcv")
INDICATORS_DIR = os.path.join(BASE_DIR, "../data/indicators")
TARGETS_DIR = os.path.join(BASE_DIR, "../data/targets")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "../data/predictions")
BACKTESTING_DIR = os.path.join(BASE_DIR, "../data/backtesting")
MODELS_DIR = os.path.join(BASE_DIR, "../models")

# Binance API Constants
SPOT_BASE_URL = "https://api.binance.com/api/v3/klines"
FUTURES_BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT","XRPUSDT"] # Add more symbols as needed: "SOLUSDT", "XRPUSDT"
START_DATE = "2023-01-01"
END_DATE = "2025-03-26"
# TODO: Test other timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d
# Note: "5m" timeframe does not work for this strategy
TIMEFRAMES = ["1h"]  # Add more timeframes as needed
MARKETS = ["futures"] # "spot", 

# Backtesting Constants
STOPLOSS = 0.01
TAKEPROFIT = 0.02

# Models used
MODEL_NAMES = ["xgboost", "catboost"]  # Add more model names as needed