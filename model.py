import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pickle

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sp500_model.pkl")

# ── 1. DATA ──────────────────────────────────────────────────────────────────

def load_data():
    sp500 = yf.Ticker("^GSPC").history(period="max")
    sp500.index = sp500.index.tz_localize(None)
    sp500 = sp500.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
    sp500 = sp500.loc["1998-01-01":].copy()

    vix = yf.Ticker("^VIX").history(period="max")[["Close"]]
    vix.columns = ["VIX"]
    vix.index = vix.index.tz_localize(None)

    sp500 = sp500.join(vix, how="left")
    sp500["VIX"] = sp500["VIX"].ffill()

    return sp500

# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────

def engineer_features(sp500):
    sp500 = sp500.copy()

    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"]   = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

    horizons   = [2, 5, 60, 250, 1000]
    predictors = ["VIX"]

    for i in horizons:
        rolling_averages = sp500.rolling(i).mean()

        col_ratio = f"Close ratio (prev {i} days)"
        sp500[col_ratio] = sp500["Close"] / rolling_averages["Close"]

        col_trend = f"Trend (prev {i} days)"
        sp500[col_trend] = sp500.shift(1).rolling(i).sum()["Target"]

        predictors += [col_ratio, col_trend]

    sp500 = sp500.dropna()
    return sp500, predictors

# ── 3. PREDICT + BACKTEST ────────────────────────────────────────────────────

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = (model.predict_proba(test[predictors])[:, 1] > 0.6).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    return pd.concat([test["Target"], preds], axis=1)

def backtesting(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test  = data.iloc[i:i+step].copy()
        all_predictions.append(predict(train, test, predictors, model))
    return pd.concat(all_predictions)

# ── 4. TRAIN + SAVE ──────────────────────────────────────────────────────────

def train_and_save(path=None):
    if path is None:
        path = MODEL_PATH

    print("Downloading data...")
    sp500 = load_data()

    print("Engineering features...")
    sp500, predictors = engineer_features(sp500)

    model = RandomForestClassifier(
        n_estimators=200, min_samples_split=50, random_state=1
    )

    print("Backtesting...")
    predictions = backtesting(sp500, model, predictors)
    score = precision_score(predictions["Target"], predictions["Predictions"])
    print(f"Backtest precision: {score:.4f}")

    # Final fit on ALL data for live prediction
    model.fit(sp500[predictors], sp500["Target"])

    with open(path, "wb") as f:
        pickle.dump({
            "model":       model,
            "predictors":  predictors,
            "predictions": predictions,
            "precision":   score
        }, f)

    print(f"Model saved to {path}")
    return model, predictors, predictions, score

# ── 5. LOAD ──────────────────────────────────────────────────────────────────

def load_model(path=None):
    if path is None:
        path = MODEL_PATH
    with open(path, "rb") as f:
        return pickle.load(f)

# ── 6. LIVE PREDICTION ───────────────────────────────────────────────────────

def predict_today(model, sp500, predictors):
    latest = sp500[predictors].iloc[-1:]
    prob   = model.predict_proba(latest)[0][1]

    if prob >= 0.6:
        label = "UP 📈"
    elif prob <= 0.4:
        label = "DOWN 📉"
    else:
        label = "UNCERTAIN ⚠️"

    return label, round(prob, 4)


if __name__ == "__main__":
    train_and_save()