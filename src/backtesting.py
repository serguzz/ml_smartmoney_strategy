import os
import pandas as pd
from config import SYMBOLS, PREDICTIONS_DIR, STOPLOSS, TAKEPROFIT
from config import MODEL_NAMES

# Constants
PREDICTION_THRESHOLDS = {
    "long": 0.50,
    "short": 0.50
}

TAKER_FEE = 0.001  # Example exchange fee (0.1%)
MAKER_FEE = 0.0005  # Example exchange fee (0.05%)

# Backtesting Results
results = []

def backtest(symbol):
    """
    Backtest the model predictions for a given symbol.
    This function calculates the number of trades, win trades, loss trades,
    and total outcome based on the model predictions.
    It iterates through the predictions for both spot and futures markets,
    and for each model, it calculates the performance metrics.    
    """
    for market in ["spot", "futures"]:
        predictions_subdir = os.path.join(PREDICTIONS_DIR, market)
        os.makedirs(predictions_subdir, exist_ok=True)
        
        for direction in ["long", "short"]:
            for model_name in MODEL_NAMES:
                file_path = f"{predictions_subdir}/{model_name}_{direction}_{symbol}_predictions.csv"
                if not os.path.exists(file_path):
                    print(f"No predictions found for {symbol}.")
                    return
                
                df = pd.read_csv(file_path)
                # Ensure the model probability column exists
                if f"{model_name}_proba" not in df.columns:
                    print(f"{model_name} probability column not found in {file_path}.")
                    return
                # Ensure the actual column exists
                if "actual" not in df.columns:
                    print(f"Actual column not found in {file_path}.")
                    return
                # Iterate through the DataFrame rows
                # Calculate trades and outcomes
                num_trades = win_trades = loss_trades = total_outcome = 0
                for _, row in df.iterrows():
                    actual = row["actual"]
                    model_proba = row[f"{model_name}_proba"]
                    if model_proba > PREDICTION_THRESHOLDS[direction]:
                        num_trades += 1
                        if actual == 1:
                            total_outcome += (TAKEPROFIT - MAKER_FEE)
                            win_trades += 1
                        else:
                            total_outcome -= (STOPLOSS + TAKER_FEE)
                            loss_trades += 1                
                # Store results
                results.append({"market": market, "model": model_name, "direction": direction, "symbol": symbol, "num_trades": num_trades, "win_trades": win_trades, "total_outcome": total_outcome})

def backtest_all():
    print("\nStarting backtesting...")
    # Run backtesting for each symbol
    for symbol in SYMBOLS:
        backtest(symbol)

    # Summary Results
    print("Backtesting Summary:")
    for res in results:
        print(f"Market: {res['market']}, Model: {res['model']}, {res['direction']} {res['symbol']} -  Tot Profit: {res['total_outcome']:.2f}, Trades: {res['num_trades']}, Wins: {res['win_trades']}, Win%: {res['win_trades'] / res['num_trades'] * 100:.1f}%")

if __name__ == "__main__":
    backtest_all()
    # Note: The backtest_all() function is called in the main block.
    # The backtest() function is called for each symbol in the SYMBOLS list.