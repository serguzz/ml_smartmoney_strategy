import os
import pandas as pd
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import pickle
import numpy as np

from config import TECHNICALS_DIR
# from indicators import shift_growth_cols

# Directories
MODELS_DIR = "models"
PREDICTIONS_DIR = "predictions"

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Features & Target
FEATURE_COLS = ["ema50_100_bullish", "ema50_200_bullish",
                "rsi_overbought", "rsi_oversold", 
                "bullish_trend_50", "bullish_trend_100", "bullish_trend_200",
                "bos", "choch"]
shift_growth_cols = []
for i in range(1, 6):
    shift_growth_cols.append(f'shift_{i}_growth')
# Add shift growth columns to feature columns
FEATURE_COLS += shift_growth_cols

# Define the target function
def add_target(df, future_window=20, up_threshold=0.02, down_threshold=0.01):
    target = np.zeros(len(df))  # Default to 0

    for i in range(len(df)):
        close_price = df.loc[i, 'close']
        high_threshold = close_price * (1 + up_threshold)
        low_threshold = close_price * (1 - down_threshold)
        
        future_highs = df.loc[i+1:i+future_window, 'high']
        future_lows = df.loc[i+1:i+future_window, 'low']

        # Find the first index where high exceeds the 2% threshold
        high_reach_idx = (future_highs >= high_threshold).idxmax() if (future_highs >= high_threshold).any() else None

        if high_reach_idx is not None:
            # Check if low falls below -1% threshold before reaching the high target
            low_drops = (future_lows.loc[i+1:high_reach_idx] <= low_threshold).any()
            if not low_drops:
                target[i] = 1    
    df['target_long'] = target
    # TODO: Add target_short logic
    return df

# Function to train models
def train_models():
    print("\nTraining models...")

    # Training long and short models for TARGET_LONG and TARGET_SHORT respectively
    # TODO: for direction in ["long", "short"]:
    for direction in ["long"]:      
        # Define target column based on direction
        target_col = f"target_{direction}"
        # Load data and train models
        for filename in os.listdir(TECHNICALS_DIR):
            if filename.endswith(".csv"):
                pair = filename.replace(".csv", "")
                print(f"Training models for {pair}...")
                
                # Load dataset
                df = pd.read_csv(os.path.join(TECHNICALS_DIR, filename))
                df.dropna(inplace=True)

                df = add_target(df)  # Ensure target is added
                
                X = df[FEATURE_COLS]
                y = df[target_col]
                
                # Split data (80% train, 20% test)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                
                # Train XGBoost Model
                xgb_model = xgb.XGBClassifier(max_depth=5, reg_lambda=20, eval_metric='logloss')
                xgb_model.fit(X_train, y_train)
                
                # Train CatBoost Model
                cat_model = cb.CatBoostClassifier(l2_leaf_reg=20, depth=5, verbose=0)
                cat_model.fit(X_train, y_train)
                
                # Save models
                with open(os.path.join(MODELS_DIR, f"{direction}_{pair}_xgb.pkl"), "wb") as f:
                    pickle.dump(xgb_model, f)
                with open(os.path.join(MODELS_DIR, f"{direction}_{pair}_cat.pkl"), "wb") as f:
                    pickle.dump(cat_model, f)
                
                # Predict probabilities on train set
                xgb_probs_train = xgb_model.predict_proba(X_train)[:, 1]
                cat_probs_train = cat_model.predict_proba(X_train)[:, 1] 

                # Predict probabilities on test set
                xgb_probs_test = xgb_model.predict_proba(X_test)[:, 1]  # Probability of target=1
                cat_probs_test = cat_model.predict_proba(X_test)[:, 1]
                
                # Evaluate Precision on train set
                avg_precision_xgb_train = average_precision_score(y_train, xgb_probs_train)
                avg_precision_cat_train = average_precision_score(y_train, cat_probs_train)

                # Evaluate Precision on test set
                avg_precision_xgb_test = average_precision_score(y_test, xgb_probs_test)
                avg_precision_cat_test = average_precision_score(y_test, cat_probs_test)
                            
                print(f"{pair} {direction} - XGB Avg Precision train: {avg_precision_xgb_train:.4f}")
                print(f"{pair} {direction} - CatBoost Avg Precision train: {avg_precision_cat_train:.4f}")
                print(f"{pair} {direction} - XGB Avg Precision test: {avg_precision_xgb_test:.4f}")
                print(f"{pair} {direction} - CatBoost Avg Precision test: {avg_precision_cat_test:.4f}")
        
                # Save predictions
                predictions_df = pd.DataFrame({
                    "actual": y_test,
                    "xgb_proba": xgb_probs_test,
                    "cat_proba": cat_probs_test
                })
                predictions_df.to_csv(os.path.join(PREDICTIONS_DIR, f"{direction}_{pair}_predictions.csv"), index=False)
            
    print("Training models complete! Predictions saved.")

if __name__ == "__main__":
    train_models()