import os
import shutil
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from config import INDICATORS_DIR, TARGETS_DIR, MODELS_DIR, PREDICTIONS_DIR
from config import TIMEFRAMES
from config import STOPLOSS, TAKEPROFIT
from config import MODEL_NAMES

# Ensure directories exist
for timeframe in TIMEFRAMES:
    for market in ["spot", "futures"]:
        # make models dirs
        os.makedirs(os.path.join(MODELS_DIR, timeframe, market), exist_ok=True)
        # make predictions dirs
        os.makedirs(os.path.join(PREDICTIONS_DIR, timeframe, market), exist_ok=True)
        # make targets dirs
        os.makedirs(os.path.join(TARGETS_DIR, timeframe, market), exist_ok=True)
        
# Constants
PREDICTION_WINDOW = 20  # Number of candles to predict into the future

# Features columns
FEATURE_COLS = ["ema50_100_bullish", "ema50_200_bullish",
                "rsi_overbought", "rsi_oversold", 
                "bullish_trend_50", "bullish_trend_100", "bullish_trend_200",
                "bos_bullish", "choch_bullish", "bos_bearish", "choch_bearish"]
shift_growth_cols = []
for i in range(1, 6):
    shift_growth_cols.append(f'shift_{i}_growth')
# Add shift growth columns to feature columns
FEATURE_COLS += shift_growth_cols


# Functions to add Long target column 
def add_target_long(df, future_window=PREDICTION_WINDOW, takeprofit=TAKEPROFIT, stoploss=STOPLOSS) -> DataFrame:
    """
    Adds 'target_long' column to df:
    target_long = 1 if price increases by `takeprofit` before dropping by `stoploss`
             within the next `future_window` candles.
    Otherwise, target_long = 0.
    """
    targets_long = []

    for i in range(len(df)):
        entry_price = df.loc[i, 'close']
        tp_price = entry_price * (1 + takeprofit)
        sl_price = entry_price * (1 - stoploss)

        tp_hit = False
        sl_hit = False

        for j in range(1, future_window + 1):
            if i + j >= len(df):
                break

            high = df.loc[i + j, 'high']
            low = df.loc[i + j, 'low']

            if low <= sl_price:
                sl_hit = True
                break  # SL hit before TP → fail

            if high >= tp_price:
                tp_hit = True
                break  # TP hit before SL → success

        if tp_hit and not sl_hit:
            targets_long.append(1)
        else:
            targets_long.append(0)

    df['target_long'] = targets_long
    return df

# Function to add Short target column
def add_target_short(df, future_window=PREDICTION_WINDOW, takeprofit=TAKEPROFIT, stoploss=STOPLOSS) -> DataFrame:
    """
    Adds 'target_short' column to df:
    target_short = 1 if price decreases by `takeprofit` before uprising by `stoploss`
             within the next `future_window` candles.
    Otherwise, target_short = 0.
    """
    targets_short = []

    for i in range(len(df)):
        entry_price = df.loc[i, 'close']
        tp_price = entry_price * (1 - takeprofit)
        sl_price = entry_price * (1 + stoploss)

        tp_hit = False
        sl_hit = False

        for j in range(1, future_window + 1):
            if i + j >= len(df):
                break

            high = df.loc[i + j, 'high']
            low = df.loc[i + j, 'low']

            if high >= sl_price:
                sl_hit = True
                break  # SL hit before TP → fail

            if low <= tp_price:
                tp_hit = True
                break  # TP hit before SL → success

        if tp_hit and not sl_hit:
            targets_short.append(1)
        else:
            targets_short.append(0)

    df['target_short'] = targets_short
    return df

# Function to train models
def train_models(timeframe) -> None:
    """
    Train models for the given timeframe.
    """
    # Define version for the current training session
    ############################################################################
    version = "v_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    models_changed = False

    # Training  models for Long and Short trading
    for direction in ["long", "short"]:      
        # Define target column based on direction
        target_col = f"target_{direction}"
        # Load data and train models for spot and futures (for each file)
        # Assuming the data is in INDICATORS_DIR/spot and INDICATORS_DIR/futures
        ############################################################################
        for market in ["spot", "futures"]:            
            # Create directories for models and predictions
            ############################################################################
            indicators_subdir = os.path.join(INDICATORS_DIR, timeframe, market)
            targets_subdir = os.path.join(TARGETS_DIR, timeframe, market)
            prediction_subdir = os.path.join(PREDICTIONS_DIR, timeframe, market)
            models_subdir = os.path.join(MODELS_DIR, timeframe, market)

            # Versioned subdirectory for predictions and models
            versioned_pred_subdir = os.path.join(PREDICTIONS_DIR, timeframe, f"{version}", market)
            versioned_models_subdir = os.path.join(MODELS_DIR, timeframe, f"{version}", market)
            
            # Create folders if they do not exist
            os.makedirs(models_subdir, exist_ok=True)
            os.makedirs(prediction_subdir, exist_ok=True)
            os.makedirs(targets_subdir, exist_ok=True)
            os.makedirs(versioned_pred_subdir, exist_ok=True)
            os.makedirs(versioned_models_subdir, exist_ok=True)

            # Iterate through indicators files and train models
            for filename in os.listdir(indicators_subdir):
                if filename.endswith(".csv"):
                    pair = filename.replace(".csv", "")
                    print(f"Training models for {pair} {timeframe} {market}...")
                    
                    # Load dataset
                    df = pd.read_csv(os.path.join(indicators_subdir, filename)) 
                    df.dropna(inplace=True)

                    # Add targets for Long and Short trading
                    if direction == "long":
                        df = add_target_long(df)
                    else:
                        df = add_target_short(df) 
                    # Save df with target column for future reference
                    df.to_csv(os.path.join(targets_subdir, filename), index=False)
                    
                    X = df[FEATURE_COLS]
                    y = df[target_col]
                    
                    # Split data (80% train, 20% test)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                    
                    # Define models
                    models = {
                        "xgboost": xgb.XGBClassifier(max_depth=5, reg_lambda=20, eval_metric='logloss'),
                        "catboost": cb.CatBoostClassifier(l2_leaf_reg=20, depth=5, verbose=0)
                    }
                    # Train models
                    for model_name in MODEL_NAMES:
                        model = models[model_name]
                        model.fit(X_train, y_train)

                        # TODO: Rename files to {model_name}_{pair}_{timeframe}_{direction}.pkl
                        # Define model paths
                        model_filename = f"{direction}_{pair}_{model_name}.pkl"
                        versioned_model_path = os.path.join(versioned_models_subdir, model_filename)
                        model_path = os.path.join(models_subdir, model_filename)

                        # Always save to versioned dir
                        with open(versioned_model_path, "wb") as f:
                            pickle.dump(model, f)

                        # Decide if we need to update the current model
                        needs_update = False

                        if os.path.exists(model_path):
                            with open(model_path, "rb") as f:
                                existing_model = pickle.load(f)
                            if not np.array_equal(model.predict(X_train[:100]), existing_model.predict(X_train[:100])):
                                print(f"[CHANGED] {model_name} for {pair} {direction} - updating current directory.")
                                needs_update = True
                            else:
                                print(f"[UNCHANGED] {model_name} for {pair} {direction} - skipping current directory update.")
                        else:
                            print(f"[NEW] {model_name} for {pair} {direction} - saving to current directory.")
                            needs_update = True

                        # Save to current dir if needed
                        if needs_update:
                            with open(model_path, "wb") as f:
                                pickle.dump(model, f)
                            models_changed = True

                        # Predict probabilities on train set
                        model_probs_train = model.predict_proba(X_train)[:, 1]

                        # Predict probabilities on test set
                        model_probs_test = model.predict_proba(X_test)[:, 1]  # Probability of target=1
                        
                        # Evaluate Precision on train set
                        avg_precision_model_train = average_precision_score(y_train, model_probs_train)

                        # Evaluate Precision on test set
                        avg_precision_model_test = average_precision_score(y_test, model_probs_test)
                                    
                        print(f"{pair} {direction} - {model_name} Avg Precision train: {avg_precision_model_train:.4f}")
                        print(f"{pair} {direction} - {model_name} Avg Precision test: {avg_precision_model_test:.4f}")
                
                        # Save predictions
                        predictions_df = pd.DataFrame({
                            "actual": y_test,
                            f"{model_name}_proba": model_probs_test,
                        })
                        # Save predictions to both directories (versioned and regular)
                        for dir in [prediction_subdir, versioned_pred_subdir]:
                            predictions_df.to_csv(os.path.join(dir, f"{model_name}_{direction}_{pair}_predictions.csv"), index=False)
                        
    print("Training models complete!")
    if models_changed:
        print(f"Some models have changed. Versioned models and predictions saved under {version}.")
    else:
        print("No models have changed.")
        # If no models were changed, remove the versioned directories
        shutil.rmtree(os.path.join(MODELS_DIR, timeframe, f"{version}"))
        shutil.rmtree(os.path.join(PREDICTIONS_DIR, timeframe, f"{version}"))


def train_all_timeframes_models() -> None:
    """
    Train models for all timeframes.
    """
    for timeframe in TIMEFRAMES:
        print(f"Training models for timeframe: {timeframe}")
        # Call the function to train models for the specific timeframe
        train_models(timeframe)

if __name__ == "__main__":
    train_all_timeframes_models()