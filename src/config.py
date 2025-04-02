import os

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OHLCV_DIR = os.path.join(BASE_DIR, "../data/ohlcv")
TECHNICALS_DIR = os.path.join(BASE_DIR, "../data/technicals")
TARGETS_DIR = os.path.join(BASE_DIR, "../data/targets")
PREDICTIONS_DIR = "predictions"

# Binance API Constants
BASE_URL = "https://api.binance.com/api/v3/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
INTERVAL = "1h"
START_DATE = "2023-01-01"
END_DATE = "2025-03-26"

# Ensure directories exist
os.makedirs(OHLCV_DIR, exist_ok=True)
os.makedirs(TECHNICALS_DIR, exist_ok=True)

# Backtesting Constants
STOPLOSS = 0.01
TAKEPROFIT = 0.02