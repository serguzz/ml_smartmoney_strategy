import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import regex
from config import SYMBOLS, TIMEFRAMES, MARKETS, STOPLOSS, TAKEPROFIT
from config import MODEL_NAMES, PREDICTIONS_DIR, BACKTESTING_DIR

# Create backtesting directory if it doesn't exist
os.makedirs(BACKTESTING_DIR, exist_ok=True)
for timeframe in TIMEFRAMES:
    # Create a subdirectory for each timeframe
    os.makedirs(os.path.join(BACKTESTING_DIR, timeframe), exist_ok=True)

# Constants
WIN_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.70, 0.75, 0.80, 0.85]
# WIN_THRESHOLDS = [0.70, 0.75, 0.80]

TAKER_FEE = 0.001  # Example exchange fee (0.1%)
MAKER_FEE = 0.0005  # Example exchange fee (0.05%)

MIN_TRADES_COUNT = 10 # Minimum required trades count to consider a threshold

# Read name of versioned predictions directory
# From PREDICTIONS_DIR, get the latest version name
def get_latest_version(timeframe) -> str:
    """
    Get the latest version of the predictions directory for a given timeframe.
    This function checks the PREDICTIONS_DIR for the latest versioned directory
    and returns the directory name, or '' if no directories are found.
    """
    # Get all directories in PREDICTIONS_DIR for the given timeframe
    predictions_dir = os.path.join(PREDICTIONS_DIR, timeframe)
    dirs = [d for d in os.listdir(predictions_dir) if os.path.isdir(predictions_dir)]
    # Filter out directories that don't meet the pattern v_123456778_123456
    # overall regex pattern: v_[0-9]{8}_[0-9]{6}
    dirs = [d for d in dirs if regex.match(r"^v_[0-9]{8}_[0-9]{6}$", d)]    
    # Sort directories by version number
    dirs.sort()
    # Return the latest directory name, '' if no directories found
    return dirs[-1] if dirs else ''  # else None


# Function to backtest the model predictions
# for a given symbol and threshold
# The function calculates the number of trades, win trades, loss trades,
# and total outcome based on the model predictions
# It iterates through prediction in a directory, whether current or versioned
def backtest_predictions(symbol, timeframe, market, threshold, version):
    """
    Backtest the model predictions for a given symbol.
    This function calculates the number of trades, win trades, loss trades,
    and total outcome based on the model predictions.
    It iterates through the predictions for both spot and futures markets,
    and for each model, it calculates the performance metrics.    
    """
    # Backtesting Results
    results = []
    results_df = pd.DataFrame()
    predictions_subdir = os.path.join(PREDICTIONS_DIR, timeframe, version, market)
    os.makedirs(predictions_subdir, exist_ok=True)
        
    for direction in ["long", "short"]:
        for model_name in MODEL_NAMES:
            file_path = os.path.join(predictions_subdir, f"{model_name}_{direction}_{symbol}_predictions.csv")
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
            # Append to results DataFrame
            results_df = pd.concat([results_df, pd.DataFrame([{
                "symbol": symbol, "model": model_name, "threshold": threshold, 
                "market": market, "direction": direction, 
                "num_trades": num_trades, "win_trades": win_trades, "win_percentage": win_percentage,
                "total_outcome": total_outcome
            }])], ignore_index=True)

    return results, results_df

def backtest_timeframe(timeframe) -> DataFrame:
    """
    Backtest all symbols with different thresholds.
    This function iterates through the symbols and thresholds,
    and calls the backtest function for each combination.
    It prints the summary of the backtesting results.
    """
    versions_to_backtest = ['']  # Current version / dir
    # Get the latest versioned predictions directory
    latest_version = get_latest_version(timeframe=timeframe)

    if latest_version:
        print(f"Latest version found: {latest_version}")
        versions_to_backtest.append(latest_version)
    else:
        print("No versions found. Backtesting only current version/dir.")
    
    for version in versions_to_backtest:  # Add latest version and empty string for non-versioned
        print(f"\nBacktesting version: {version} !")        
        # Iterate through each symbol and threshold
        for symbol in SYMBOLS:
            symbol_results_df = pd.DataFrame()
            for market in MARKETS:
                # Track best result for each symbol and market
                best_result = {"threshold": None, "sharpe_ratio": float('-inf'), "total_outcome": float('-inf'), "num_trades": 0, "win_trades": 0, "total_win_percentage": 0}
                for threshold in WIN_THRESHOLDS:
                    threshold_results = []
                    print(f"\nBacktesting {symbol} {timeframe} {market}, threshold {threshold}")
                    print("---------------------------------------------------")
                    results, results_df = backtest_predictions(symbol, timeframe, market, threshold, version)
                    threshold_results.extend(results)
                    symbol_results_df = pd.concat([symbol_results_df, results_df], ignore_index=True)

                    # Summary Results
                    # print("Backtesting Summary:")
                    for res in threshold_results:
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
                    print(f"Sharpe Ratio: {sharpe_ratio:.3f}, Total Profit: {total_profit:.2f}, Total Trades: {total_trades}, Total Wins: {total_wins}, Total Win%: {total_win_percentage:.1f}%\n")
                    
                    # Update best results for the symbol
                    if sharpe_ratio > best_result["sharpe_ratio"]:
                        best_result["sharpe_ratio"] = sharpe_ratio
                        best_result["threshold"] = threshold
                        best_result["total_outcome"] = total_profit
                        best_result["num_trades"] = total_trades
                        best_result["win_trades"] = total_wins
                        best_result["total_win_percentage"] = total_win_percentage

                    # if sum of trades in results is 0, skip
                    if sum(res["num_trades"] for res in results) < MIN_TRADES_COUNT:
                        print(f"Less than {MIN_TRADES_COUNT} trades executed for {symbol} with threshold {threshold}. Skippint all higher thresholds.")
                        break
                
                # Print best results for this symbol and market
                print(f"Best Results for {symbol} {market}:")
                print(f"Threshold: {best_result['threshold']}, Sharpe Ratio: {best_result['sharpe_ratio']:.2f}, Total Profit: {best_result['total_outcome']:.2f}, Total Trades: {best_result['num_trades']}, Wins: {best_result['win_trades']}, Win%: {best_result['total_win_percentage']:.1f}%")
                # print(f"{symbol}: Threshold:{res['threshold']}| Sharpe Ratio: {res['sharpe_ratio']:.2f}, Total Profit: {res['total_outcome']:.2f}, Total Trades: {res['num_trades']}, Wins: {res['win_trades']}, Win%: {res['total_win_percentage']:.1f}%")

            # Round the values for better readability            
            symbol_results_df["win_percentage"] = symbol_results_df["win_percentage"].round(2)
            symbol_results_df["total_outcome"] = symbol_results_df["total_outcome"].round(4)
            # Create a directory for the timeframe/version
            save_dir = os.path.join(BACKTESTING_DIR, timeframe, version)
            os.makedirs(save_dir, exist_ok=True)
            symbol_results_df.to_csv(os.path.join(save_dir, f"{symbol}_backtesting_results.csv"), index=False)
            # Sort results by metric: score = win_percentage * log(num_trades)
            # and save to a file
            symbol_results_df['score'] = (symbol_results_df['win_percentage'] * np.log(symbol_results_df['num_trades'] + 1)).round(1)
            symbol_results_df = symbol_results_df.sort_values(
                by=['market','direction','score'], 
                ascending=[True, True, False]
                )
            symbol_results_df.to_csv(os.path.join(save_dir, f"{symbol}_results_analyzed.csv"), index=False)

        print("---------------------------------------------------")
        print(f"Backtesting completed for {timeframe}")
            
    # Return
    return

def backtest_all_timeframes() -> None:
    """
    Backtest all timeframes.
    This function iterates through the timeframes,
    and calls the backtest_all.
    It prints the summary of the backtesting results.
    """
    for timeframe in TIMEFRAMES:
        print(f"\nStarting backtesting for timeframe: {timeframe}\n")
        backtest_timeframe(timeframe)

if __name__ == "__main__":
    backtest_all_timeframes()
    # Note: The backtest_all() function is called in the main block.
    # The backtest() function is called for each symbol in the SYMBOLS list.