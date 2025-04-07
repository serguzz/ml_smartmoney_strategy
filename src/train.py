import os
import pandas as pd
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import pickle
import numpy as np

from config import INDICATORS_DIR, TARGETS_DIR, MODELS_DIR, PREDICTIONS_DIR
from config import STOPLOSS, TAKEPROFIT
from config import MODEL_NAMES

# Ensure directories exist
os.makedirs(INDICATORS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(TARGETS_DIR, exist_ok=True)

# Features & Target
FEATURE_COLS = ["ema50_100_bullish", "ema50_200_bullish",
                "rsi_overbought", "rsi_oversold", 
                "bullish_trend_50", "bullish_trend_100", "bullish_trend_200",
                "bos_bullish", "choch_bullish", "bos_bearish", "choch_bearish"]
shift_growth_cols = []
for i in range(1, 6):
    shift_growth_cols.append(f'shift_{i}_growth')
# Add shift growth columns to feature columns
FEATURE_COLS += shift_growth_cols

def add_target_long(df, future_window=20, takeprofit=TAKEPROFIT, stoploss=STOPLOSS):
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

def add_target_short(df, future_window=20, takeprofit=TAKEPROFIT, stoploss=STOPLOSS):
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
def train_models():
    print("\nTraining models...")

    # Training long and short models for TARGET_LONG and TARGET_SHORT respectively
    # TODO: for direction in ["long", "short"]:
    for direction in ["long", "short"]:      
        # Define target column based on direction
        target_col = f"target_{direction}"
        # Load data and train models for spot and futures (for each file)
        # Assuming the data is in INDICATORS_DIR/spot and INDICATORS_DIR/futures
        ############################################################################
        for market in ["spot", "futures"]:
            # Define subdirectories for spot and futures
            ############################################################################
            indicators_subdir = os.path.join(INDICATORS_DIR, market)
            targets_subdir = os.path.join(TARGETS_DIR, market)
            prediction_subdir = os.path.join(PREDICTIONS_DIR, market)
            models_subdir = os.path.join(MODELS_DIR, market)
            os.makedirs(models_subdir, exist_ok=True)
            os.makedirs(prediction_subdir, exist_ok=True)
            os.makedirs(targets_subdir, exist_ok=True)

            # Iterate through indicators files and train models
            for filename in os.listdir(indicators_subdir):
                if filename.endswith(".csv"):
                    pair = filename.replace(".csv", "")
                    print(f"Training models for {pair} {market}...")
                    
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

                        # Save models
                        with open(os.path.join(models_subdir, f"{direction}_{pair}_{model_name}.pkl"), "wb") as f:
                            pickle.dump(model, f)

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
                        predictions_df.to_csv(os.path.join(prediction_subdir, f"{model_name}_{direction}_{pair}_predictions.csv"), index=False)


            
    print("Training models complete! Predictions saved.")

if __name__ == "__main__":
    train_models()