import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from model import load_data, engineer_features, load_model, predict_today
from news import get_today_sentiment

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="S&P 500 Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 S&P 500 Market Predictor")
st.caption("Random Forest + VIX + News Sentiment")

# ── LOAD MODEL + DATA ─────────────────────────────────────────────────────────

# @st.cache_resource
# def get_model_and_data():
#     pkg = load_model("sp500_model.pkl")
#     sp500, predictors = engineer_features(load_data())
#     return pkg, sp500, predictors

@st.cache_resource
def get_model_and_data():
    from model import train_and_save
    if not os.path.exists("sp500_model.pkl"):
        with st.spinner("Training model for first time... (this takes 2-3 mins)"):
            train_and_save()
    pkg = load_model("sp500_model.pkl")
    sp500, predictors = engineer_features(load_data())
    return pkg, sp500, predictors

with st.spinner("Loading model and data..."):
    pkg, sp500, predictors = get_model_and_data()

model       = pkg["model"]
predictions = pkg["predictions"]
precision   = pkg["precision"]

# ── SECTION 1: TODAY'S PREDICTION ────────────────────────────────────────────

st.header("🔮 Tomorrow's Prediction")

label, prob = predict_today(model, sp500, predictors)
confidence  = round(prob * 100, 1)

col1, col2, col3 = st.columns(3)

with col1:
    color = "green" if "UP" in label else "red" if "DOWN" in label else "orange"
    st.markdown(
        f"<h1 style='color:{color}; text-align:center'>{label}</h1>",
        unsafe_allow_html=True
    )
with col2:
    st.metric("Model Confidence", f"{confidence}%")
with col3:
    st.metric("Backtest Precision", f"{round(precision * 100, 1)}%",
              help="% of UP predictions that were correct during backtesting")

st.info(
    "ℹ️ The model only predicts **UP** when confidence ≥ 60%. "
    "Otherwise it abstains (shows DOWN/UNCERTAIN). "
    "Only ~460 UP calls were made across 3,600+ trading days — "
    "this is intentional conservatism, not a bug."
)

st.divider()

# ── SECTION 2: NEWS & SENTIMENT ───────────────────────────────────────────────

st.header("📰 Today's News & Sentiment")

@st.cache_data(ttl=3600)
def get_sentiment():
    return get_today_sentiment()

with st.spinner("Fetching today's headlines..."):
    sentiment = get_sentiment()

slabel   = sentiment["label"]
sscore   = sentiment["score"]
headlines = sentiment.get("headlines", [])

scol1, scol2 = st.columns([1, 2])

with scol1:
    scolor = "green" if slabel == "bullish" else "red" if slabel == "bearish" else "gray"
    st.markdown(
        f"<h2 style='color:{scolor}; text-align:center'>{slabel.upper()}</h2>",
        unsafe_allow_html=True
    )
    st.metric("Sentiment Score", f"{sscore:+.2f}",
              help="-1.0 = most bearish, +1.0 = most bullish")

with scol2:
    if headlines:
        st.markdown("**Today's Headlines:**")
        for h in headlines:
            st.markdown(f"- {h}")
    else:
        st.info("No headlines available for today yet — try again later.")

# ── COMBINED SIGNAL ───────────────────────────────────────────────────────────

st.subheader("🧠 Combined Signal")

model_bullish = prob >= 0.6
model_bearish = prob <= 0.4
news_bullish  = sscore > 0.2
news_bearish  = sscore < -0.2

if model_bullish and news_bullish:
    st.success("✅ Strong BUY signal — Model and news both bullish")
elif model_bearish and news_bearish:
    st.error("🚨 Strong SELL signal — Model and news both bearish")
elif model_bullish and news_bearish:
    st.warning("⚠️ Mixed signal — Model bullish but news bearish")
elif model_bearish and news_bullish:
    st.warning("⚠️ Mixed signal — Model bearish but news bullish")
else:
    st.info("➡️ Neutral — No strong signal either way")

st.divider()

# ── SECTION 3: VIX CHART ──────────────────────────────────────────────────────

st.header("😰 VIX — Fear & Greed Index")

@st.cache_data(ttl=3600)
def get_vix():
    vix = yf.Ticker("^VIX").history(period="3mo")[["Close"]]
    vix.index = vix.index.tz_localize(None)
    return vix

vix_df      = get_vix()
current_vix = round(vix_df["Close"].iloc[-1], 2)

fig_vix = go.Figure()
fig_vix.add_trace(go.Scatter(
    x=vix_df.index,
    y=vix_df["Close"],
    mode="lines",
    line=dict(color="orange", width=2),
    fill="tozeroy",
    fillcolor="rgba(255,165,0,0.1)",
    name="VIX"
))
fig_vix.add_hline(y=20, line_dash="dash", line_color="green",
                  annotation_text="Low Fear (<20)", annotation_position="top right")
fig_vix.add_hline(y=30, line_dash="dash", line_color="red",
                  annotation_text="High Fear (>30)", annotation_position="top right")
fig_vix.update_layout(
    title="VIX — Last 3 Months",
    xaxis_title="Date",
    yaxis_title="VIX",
    height=400,
    template="plotly_dark",
    xaxis=dict(range=[vix_df.index.min(), vix_df.index.max()])
)
st.plotly_chart(fig_vix, use_container_width=True)

if current_vix > 30:
    st.error(f"VIX is HIGH at {current_vix} — market is fearful, volatility elevated")
elif current_vix < 20:
    st.success(f"VIX is LOW at {current_vix} — market is calm")
else:
    st.info(f"VIX is MODERATE at {current_vix}")

st.divider()

# ── SECTION 4: BACKTEST RESULTS ───────────────────────────────────────────────

st.header("🔬 Backtest Results")

total    = len(predictions)
up_preds = int(predictions["Predictions"].sum())
correct  = int(((predictions["Predictions"] == 1) & (predictions["Target"] == 1)).sum())

bcol1, bcol2, bcol3 = st.columns(3)
with bcol1:
    st.metric("Total Trading Days Tested", f"{total:,}")
with bcol2:
    st.metric("UP Predictions Made", f"{up_preds:,}",
              help="Model only calls UP when ≥60% confident — very conservative by design")
with bcol3:
    st.metric("Precision Score", f"{round(precision * 100, 1)}%")

# ── DATE LOOKUP ───────────────────────────────────────────────────────────────

st.subheader("🗓️ Look Up a Specific Date")

min_date = predictions.index.min().date()
max_date = predictions.index.max().date()

selected_date = st.date_input(
    "Select a date to see what the model predicted:",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

selected_ts = pd.Timestamp(selected_date)

# Find closest available trading day
if selected_ts in predictions.index:
    row = predictions.loc[selected_ts]
    display_date = selected_ts
else:
    nearest_idx = predictions.index.get_indexer([selected_ts], method="nearest")[0]
    display_date = predictions.index[nearest_idx]
    row = predictions.loc[display_date]
    st.caption(
        f"No trading data for {selected_date} (weekend/holiday) — "
        f"showing nearest trading day: {display_date.date()}"
    )

pred_label   = "UP 📈" if row["Predictions"] == 1 else "DOWN/ABSTAIN 📉"
actual_label = "UP 📈" if row["Target"] == 1 else "DOWN 📉"
is_correct   = int(row["Predictions"]) == int(row["Target"])

dcol1, dcol2, dcol3 = st.columns(3)

with dcol1:
    pcolor = "green" if row["Predictions"] == 1 else "red"
    st.markdown("**Model Predicted:**")
    st.markdown(f"<h3 style='color:{pcolor}'>{pred_label}</h3>", unsafe_allow_html=True)

with dcol2:
    acolor = "green" if row["Target"] == 1 else "red"
    st.markdown("**What Actually Happened:**")
    st.markdown(f"<h3 style='color:{acolor}'>{actual_label}</h3>", unsafe_allow_html=True)

with dcol3:
    st.markdown("**Result:**")
    if row["Predictions"] == 0:
        st.markdown("<h3 style='color:gray'>Model abstained</h3>", unsafe_allow_html=True)
        st.caption("Confidence was below 60% — no UP call made")
    elif is_correct:
        st.markdown("<h3 style='color:green'>✅ Correct</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:red'>❌ Wrong</h3>", unsafe_allow_html=True)

# ── HEADLINES FOR SELECTED DATE (always shown, not nested in correct/wrong) ───

st.markdown("---")
scored_path = "data/headlines_scored.csv"

if os.path.exists(scored_path):
    scored_df = pd.read_csv(scored_path)
    scored_df["date"] = pd.to_datetime(scored_df["date"]).dt.strftime("%Y-%m-%d")
    date_str = display_date.strftime("%Y-%m-%d")
    match = scored_df[scored_df["date"] == date_str]

    if not match.empty:
        row_news = match.iloc[0]
        ncol1, ncol2 = st.columns([1, 2])

        with ncol1:
            nlabel = row_news["sentiment_label"]
            nscore = float(row_news["sentiment_score"])
            ncolor = "green" if nlabel == "bullish" else "red" if nlabel == "bearish" else "gray"
            st.markdown("**News Sentiment that day:**")
            st.markdown(
                f"<h3 style='color:{ncolor}'>{nlabel.upper()}</h3>",
                unsafe_allow_html=True
            )
            st.metric("Sentiment Score", f"{nscore:+.2f}")

        with ncol2:
            st.markdown("**Headlines that day:**")
            for headline in str(row_news["headlines"]).split(" | "):
                st.markdown(f"- {headline}")
    else:
        st.info("📭 No headlines cached for this date — only last 30 days available from NewsAPI free tier.")
else:
    st.info("No scored headlines file found.")

# ── ROLLING PRECISION CHART ───────────────────────────────────────────────────

up_only = predictions[predictions["Predictions"] == 1].copy()
up_only["correct"] = (up_only["Target"] == 1).astype(int)
rolling_precision = up_only["correct"].rolling(250).mean() * 100

fig_bt = go.Figure()
fig_bt.add_trace(go.Scatter(
    x=rolling_precision.index,
    y=rolling_precision.values,
    mode="lines",
    line=dict(color="cyan", width=2),
    name="Rolling Precision (1yr window)"
))
fig_bt.add_hline(y=54, line_dash="dash", line_color="gray",
                 annotation_text="Blind benchmark (54%)",
                 annotation_position="top right")
fig_bt.update_layout(
    title="Rolling 1-Year Precision Over Time",
    xaxis_title="Date",
    yaxis_title="Precision %",
    height=400,
    template="plotly_dark",
    xaxis=dict(range=[predictions.index.min(), predictions.index.max()]),
    yaxis=dict(range=[40, 70])
)
st.plotly_chart(fig_bt, use_container_width=True)

# ── RAW TABLE ─────────────────────────────────────────────────────────────────

with st.expander("📋 View Raw Predictions Table"):
    st.dataframe(
        predictions.tail(100).sort_index(ascending=False),
        use_container_width=True
    )

st.divider()
st.caption("⚠️ This is not financial advice. For educational purposes only.")