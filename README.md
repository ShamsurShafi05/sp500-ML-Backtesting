# 📈 S&P 500 Market Predictor

A machine learning system that predicts next-day S&P 500 movement using Random Forest, VIX volatility data, and real-time news sentiment analysis.

**Live Demo:** [https://sp500-ml-backtesting-hjpb5t6ukterxfkgmhtvyh.streamlit.app/](https://sp500-ml-backtesting-hjpb5t6ukterxfkgmhtvyh.streamlit.app/)

---

## Key Highlights

- **Multi-signal prediction:** Combines technical features, market fear index (VIX), and LLM-scored news sentiment
- **Rigorous backtesting:** Walk-forward backtesting across 3,600+ trading days to prevent data leakage
- **Live sentiment pipeline:** Fetches daily S&P 500 headlines via NewsAPI and scores them bullish/bearish/neutral using Groq (Llama 3.3)
- **Honest evaluation:** Model only predicts UP when confidence ≥ 60% — conservative by design, not a bug
- **Deployed:** Fully live on Streamlit Cloud with interactive charts and date-based prediction lookup

---

## Overview

The app predicts whether the S&P 500 will go **UP or DOWN** the next trading day by combining three signal sources:

- **Historical price features** — rolling close ratios and trend counts across 2, 5, 60, 250, and 1000-day horizons
- **VIX (Volatility Index)** — market fear gauge fetched live from Yahoo Finance
- **News sentiment** — daily headlines scored by Llama 3.3 via Groq API, producing a sentiment label and score from -1.0 (bearish) to +1.0 (bullish)

---

## What the App Shows

- **Tomorrow's Prediction** — UP / DOWN / UNCERTAIN with confidence percentage
- **Today's News & Sentiment** — live headlines + Groq sentiment score + combined signal
- **VIX Chart** — interactive 3-month volatility chart with fear/calm thresholds
- **Backtest Results** — precision score, rolling 1-year precision chart, and per-date prediction lookup with headlines

---

## How It Works

### 1. Data
S&P 500 price history (1998–present) and VIX history fetched live from Yahoo Finance via `yfinance`. No static dataset files — always up to date.

### 2. Feature Engineering
For each of 5 time horizons (2, 5, 60, 250, 1000 days):
- **Close Ratio** — today's price vs rolling average (captures momentum)
- **Trend** — how many of the last N days saw price rise (captures directional bias)

VIX is added as an additional predictor.

### 3. Model
Random Forest Classifier (`n_estimators=200`, `min_samples_split=50`) with a **0.6 probability threshold** for UP predictions — only predicts UP when genuinely confident, otherwise abstains.

### 4. Backtesting
Walk-forward backtesting starting from day 2500, rolling forward in 250-day steps. Each window trains only on past data and tests on unseen future data — no data leakage.

### 5. News Sentiment Pipeline
```
NewsAPI → fetch daily S&P 500 headlines
    ↓
Groq (Llama 3.3) → score as bullish / bearish / neutral + score (-1.0 to +1.0)
    ↓
Cache to CSV → display in app + use as confidence filter
```

Headlines are cached locally so API calls are minimised. Sentiment acts as a **post-prediction confidence filter** rather than a training feature (NewsAPI free tier only provides 30 days of history).

### 6. Combined Signal Logic
| Model | News | Signal |
|-------|------|--------|
| Bullish | Bullish | ✅ Strong BUY |
| Bearish | Bearish | 🚨 Strong SELL |
| Bullish | Bearish | ⚠️ Mixed |
| Bearish | Bullish | ⚠️ Mixed |
| Either | Neutral | ➡️ Neutral |

---

## Results

| Metric | Value |
|--------|-------|
| Backtest precision | ~52.7% |
| Blind benchmark (buy every day) | ~54% |
| Total days tested | 3,649 |
| UP predictions made | ~463 |

The model is intentionally conservative — it makes very few UP calls (only ~13% of days) but those calls are evaluated at 52.7% precision. The low call rate means it avoids noise rather than guessing constantly.

---

## Project Structure

```
sp500-ml-backtesting/
├── app.py                  # Streamlit app
├── model.py                # Data loading, feature engineering, training, backtesting
├── news.py                 # NewsAPI fetch + Groq sentiment scoring
├── data/
│   ├── headlines.csv           # Cached raw headlines (not committed)
│   └── headlines_scored.csv    # Cached sentiment scores (not committed)
├── sp500_model.pkl         # Trained model (not committed, generated on first run)
├── .env                    # API keys (never committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Technologies Used

- **Python 3.12**
- **Streamlit** — web app framework
- **yfinance** — S&P 500 and VIX data
- **scikit-learn** — Random Forest, precision scoring
- **NewsAPI** — daily financial headlines
- **Groq (Llama 3.3-70b)** — LLM sentiment scoring
- **Plotly** — interactive charts
- **Pandas / NumPy** — data manipulation

---

## Local Installation

### Prerequisites
- Python 3.8+
- NewsAPI key (free at [newsapi.org](https://newsapi.org))
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/ShamsurShafi05/sp500-ML-Backtesting.git
cd sp500-ML-Backtesting
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create `.env` file**
```
NEWSAPI_KEY=your_newsapi_key_here
GROQ_API_KEY=your_groq_api_key_here
```

5. **Train the model** (one-time, ~2-3 minutes)
```bash
python model.py
```

6. **Fetch and score headlines**
```bash
python news.py
```

7. **Run the app**
```bash
streamlit run app.py
```

---

## Deployment (Streamlit Cloud)

The app is configured to train the model automatically on first run if no `.pkl` file is found. To deploy your own instance:

1. Push to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set your secrets under **Manage App → Secrets**:
```toml
NEWSAPI_KEY = "your_key_here"
GROQ_API_KEY = "your_key_here"
```

---

## Important Notes

- - **NewsAPI free tier** — 100 requests/day, 30 days of history. Headlines and sentiment scores are cached to `data/headlines_scored.csv` and committed to the repo. This file must be manually refreshed monthly as older headlines fall outside the free tier's 30-day window.
- **First load** — model trains fresh from Yahoo Finance (~2-3 minutes). Subsequent loads use the cached model.
- **Sentiment vs training** — news sentiment is used as a confidence filter at prediction time, not as a training feature, since historical headline data is not available on the free tier.

---

## Future Scopes

- [ ] Add global indices as predictors (Nikkei `^N225`, FTSE `^FTSE`, DAX `^GDAXI`) — test if overnight Asian/European market moves predict S&P 500 opens
- [ ] Swap Random Forest for XGBoost or LightGBM and compare precision scores
- [ ] Add email or Telegram alerts when the model predicts UP with high confidence
- [ ] Incorporate macro indicators (Fed interest rates, CPI) via FRED API
- [ ] Expand to hourly data for higher resolution predictions
- [ ] Build a hybrid model combining content-based and collaborative filtering approaches

---

## Acknowledgements

- [Yahoo Finance](https://finance.yahoo.com) via `yfinance` for market data
- [NewsAPI](https://newsapi.org) for financial headlines
- [Groq](https://groq.com) for fast LLM inference
- [Streamlit](https://streamlit.io) for the web framework

---

## Disclaimer

⚠️ This project is for **educational purposes only**. Nothing in this app constitutes financial advice. Do not make investment decisions based on model outputs.

---

**Built with Python, Random Forest, and a lot of market data 📊**
