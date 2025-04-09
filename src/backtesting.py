import os
import numpy as np
import pandas as pd
from config import SYMBOLS, PREDICTIONS_DIR, BACKTESTING_DIR, STOPLOSS, TAKEPROFIT
from config import MODEL_NAMES

# Create backtesting directory if it doesn't exist
os.makedirs(BACKTESTING_DIR, exist_ok=True)

# Constants
WIN_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.70, 0.75, 0.80, 0.85]

TAKER_FEE = 0.001  # Example exchange fee (0.1%)
MAKER_FEE = 0.0005  # Example exchange fee (0.05%)

# Read name of versioned predictions directory
# From PREDICTIONS_DIR, get the latest version name
def get_latest_version():
    # Get all directories in PREDICTIONS_DIR
    dirs = [d for d in os.listdir(PREDICTIONS_DIR) if os.path.isdir(PREDICTIONS_DIR)]
    # Sort directories by version number
    dirs.sort()
    # Return the latest directory name, '' if no directories found
    return dirs[-1] if dirs else ''  # else None


# Function to backtest the model predictions
# for a given symbol and threshold
# The function calculates the number of trades, win trades, loss trades,
# and total outcome based on the model predictions
# It iterates through prediction in a directory, whether current or versioned
def backtest(symbol, threshold=0.50, version=''):
    """
    Backtest the model predictions for a given symbol.
    This function calculates the number of trades, win trades, loss trades,
    and total outcome based on the model predictions.
    It iterates through the predictions for both spot and futures markets,
    and for each model, it calculates the performance metrics.    
    """
    # Backtesting Results
    results = []
    for market in ["spot", "futures"]:
        predictions_subdir = os.path.join(PREDICTIONS_DIR, version, market)
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
                    if model_proba > threshold:
                        num_trades += 1
                        if actual == 1:
                            total_outcome += (TAKEPROFIT - MAKER_FEE)
                            win_trades += 1
                        else:
                            total_outcome -= (STOPLOSS + TAKER_FEE)
                            loss_trades += 1                
                win_percentage = (win_trades / num_trades * 100) if num_trades > 0 else 0
                # Store results
                results.append({
                    "symbol": symbol, "model": model_name, "threshold": threshold, 
                    "market": market, "direction": direction, 
                    "num_trades": num_trades, "win_trades": win_trades, "win_percentage": win_percentage,
                    "total_outcome": total_outcome
                    })
    return results

def backtest_all():
    """
    Backtest all symbols with different thresholds.
    This function iterates through the symbols and thresholds,
    and calls the backtest function for each combination.
    It prints the summary of the backtesting results.
    """
    print("\nStarting backtesting...")
    # Get the latest versioned predictions directory
    latest_version = get_latest_version()
    # create the versioned backtesting directory
    versioned_backtesting_dir = os.path.join(BACKTESTING_DIR, latest_version)
    os.makedirs(versioned_backtesting_dir, exist_ok=True)

    for version in ['', latest_version]:  # Add latest version and empty string for non-versioned
        
        all_symbols_results = []
        # Best threshold and results for each symbol
        best_results = {}
        for symbol in SYMBOLS:
            best_results[symbol] = {"threshold": None, "sharpe_ratio": float('-inf'), "total_outcome": float('-inf'), "num_trades": 0, "win_trades": 0, "total_win_percentage": 0}
        
        # Iterate through each symbol and threshold
        for symbol in SYMBOLS:
            for threshold in WIN_THRESHOLDS:
                all_results = []
                results = backtest(symbol, threshold, version)
                all_results.extend(results)
                all_symbols_results.extend(results)
                print(f"Backtesting symbol {symbol} with threshold {threshold}...")

                # Summary Results
                # print("Backtesting Summary:")
                for res in all_results:
                    if res["num_trades"] > 0:
                        print(f"Market: {res['market']}, Model: {res['model']}, {res['direction']} {res['symbol']} -  Tot Profit: {res['total_outcome']:.2f}, Trades: {res['num_trades']}, Wins: {res['win_trades']}, Win%: {res['win_trades'] / res['num_trades'] * 100:.1f}%")
                    else:
                        print(f"Market: {res['market']}, Model: {res['model']}, {res['direction']} {res['symbol']} - No trades executed.")

                total_profit = sum(res["total_outcome"] for res in results)
                total_trades = sum(res["num_trades"] for res in results)
                total_wins = sum(res["win_trades"] for res in results)
                total_win_percentage = (total_wins / total_trades * 100) if total_trades > 0 else 0
                
                # Calculate Sharpe Ratio
                mean_return = np.mean([res["total_outcome"] for res in results])
                std_return = np.std([res["total_outcome"] for res in results])
                sharpe_ratio = mean_return / std_return if std_return != 0 else 0
                print("---------------------------------------------------")
                print(f"Sharpe Ration: {sharpe_ratio:.3f}, Total Profit: {total_profit:.2f}, Total Trades: {total_trades}, Total Wins: {total_wins}, Total Win%: {total_win_percentage:.1f}%\n")
                
                # Update best results for the symbol
                # if total_profit > best_results[symbol]["total_outcome"]:
                if sharpe_ratio > best_results[symbol]["sharpe_ratio"]:
                    best_results[symbol]["sharpe_ratio"] = sharpe_ratio
                    best_results[symbol]["threshold"] = threshold
                    best_results[symbol]["total_outcome"] = total_profit
                    best_results[symbol]["num_trades"] = total_trades
                    best_results[symbol]["win_trades"] = total_wins
                    best_results[symbol]["total_win_percentage"] = total_win_percentage
        # Print best results for each symbol
        print("\nBest Results:")
        for symbol, res in best_results.items():
            print(f"{symbol}: Threshold:{res['threshold']}| Sharpe Ratio: {res['sharpe_ratio']:.2f}, Total Profit: {res['total_outcome']:.2f}, Total Trades: {res['num_trades']}, Wins: {res['win_trades']}, Win%: {res['total_win_percentage']:.1f}%")
        print("---------------------------------------------------")
        print("Backtesting completed.")
        # Save results to CSV to backtesting directory
        all_results_df = pd.DataFrame(all_symbols_results)
        # Round the values for better readability
        all_results_df["win_percentage"] = all_results_df["win_percentage"].round(2)
        all_results_df["total_outcome"] = all_results_df["total_outcome"].round(4)
        all_results_df.to_csv(os.path.join(BACKTESTING_DIR, version, "backtesting_results.csv"), index=False)
        print(f"All backtesting results saved to {version}/backtesting_results.csv\n")

if __name__ == "__main__":
    backtest_all()
    # Note: The backtest_all() function is called in the main block.
    # The backtest() function is called for each symbol in the SYMBOLS list.