# sp500-ML-Backtesting

Predicting whether the S&P 500 will close **higher or lower the next day** using a Random Forest classifier with a rolling backtesting system.

> **Final precision: ~55%** vs 54% blind-buy benchmark

---

## Overview

Instead of predicting exact prices, this project frames the problem as binary classification — will tomorrow's closing price be higher than today's? A backtesting system simulates realistic trading conditions by training only on past data before making each prediction.

## How it works

**Feature engineering** — five rolling time horizons (2, 5, 60, 250, 1000 days):
- Close price ratios (today's price vs. rolling average)
- Trend scores (how many times price rose in the past N days)

**Backtesting** — walk-forward validation:
- Train on the first 10 years (~2500 trading days)
- Predict the next year (~250 days)
- Expand the training window and repeat

**Model** — `RandomForestClassifier` with a 0.6 probability threshold (only predicts "up" when confident)

## Results

| Model | Precision |
|---|---|
| Blind buy every day (benchmark) | ~54% |
| Baseline Random Forest | ~52% |
| Tuned RF + feature engineering | **~55%** |

## Stack

- `yfinance` — S&P 500 historical data via Yahoo Finance
- `pandas` / `numpy` — data processing
- `scikit-learn` — Random Forest, precision scoring
- `matplotlib` — visualizations

## Setup

```bash
pip install -r requirements.txt
```

Then open `stock_market_prediction_using_random_forest.ipynb` in Jupyter or Google Colab.

## Known Limitations & Future Scopes

- Only uses price/volume data — no macro indicators, news sentiment, or sector data
- Daily resolution; intraday patterns are ignored
- Overnight markets (e.g. Nikkei, FTSE) may correlate with S&P500 moves
- Adding interest rates, inflation, and key sector performance could improve accuracy
