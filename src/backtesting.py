import os
import pandas as pd
from config import SYMBOLS, PREDICTIONS_DIR

# Constants
LONG_THRESHOLD = 0.55
SHORT_THRESHOLD = 0.5
OUTCOME_LONG = 0.02
OUTCOME_SHORT = 0.01

# Backtesting Results
results = []

def backtest(symbol):

#   TODO: for direction in ["long", "short"]:
    for direction in ["long"]:
        win_threshold = LONG_THRESHOLD if direction == "long" else SHORT_THRESHOLD
        file_path = f"{PREDICTIONS_DIR}/{direction}_{symbol}_predictions.csv"
        if not os.path.exists(file_path):
            print(f"No predictions found for {symbol}.")
            return
        
        df = pd.read_csv(file_path)
        num_trades = {
            'xgb': 0,
            'cat': 0
            }
        total_outcome = {
            'xgb': 0,
            'cat': 0
            }

        for model_name in ["xgb", "cat"]:
            for _, row in df.iterrows():
                actual = row["actual"]
                model_proba = row[f"{model_name}_proba"]
                if model_proba > win_threshold:
                    num_trades[model_name] += 1
                    if direction == "long":
                        if actual == 1:
                            total_outcome[model_name] += OUTCOME_LONG
                        else:
                            total_outcome[model_name] -= OUTCOME_SHORT
                    else:
                        if actual == 1:
                            total_outcome[model_name] += OUTCOME_SHORT
                        else:
                            total_outcome[model_name] -= OUTCOME_LONG

            results.append({"model": model_name, "direction": direction, "symbol": symbol, "num_trades": num_trades[model_name], "total_outcome": total_outcome[model_name]})

def backtest_all():
    print("\nStarting backtesting...")
    # Run backtesting for each symbol
    for symbol in SYMBOLS:
        backtest(symbol)

    # Summary Results
    print("Backtesting Summary:")
    for res in results:
        print(f"Model: {res['model']}, {res['direction']} {res['symbol']} - Trades: {res['num_trades']}, Total Outcome: {res['total_outcome']:.4f}")

if __name__ == "__main__":
    backtest_all()
    # Note: The backtest_all() function is called in the main block.
    # The backtest() function is called for each symbol in the SYMBOLS list.