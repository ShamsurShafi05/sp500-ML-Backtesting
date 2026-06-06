import os
import json
import pandas as pd
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── 1. FETCH HEADLINES ────────────────────────────────────────────────────────

def fetch_headlines(date_str):
    """Fetch up to 5 S&P500 headlines for a given date (format: YYYY-MM-DD)"""
    try:
        articles = newsapi.get_everything(
            q="S&P 500 OR stock market OR economy",
            from_param=date_str,
            to=date_str,
            language="en",
            sort_by="relevancy",
            page_size=5
        )
        return [a["title"] for a in articles["articles"]]
    except Exception as e:
        print(f"NewsAPI error for {date_str}: {e}")
        return []

def fetch_and_cache_headlines(path="data/headlines.csv", days=30):
    """Fetch last 30 days of headlines, skip dates already cached"""
    os.makedirs("data", exist_ok=True)

    # Load existing cache
    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing["date"] = pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d")
        cached_dates = set(existing["date"].tolist())
    else:
        existing = pd.DataFrame(columns=["date", "headlines"])
        cached_dates = set()

    rows = []
    for i in range(days):
        date = (datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d")
        if date in cached_dates:
            continue
        headlines = fetch_headlines(date)
        if headlines:
            rows.append({"date": date, "headlines": " | ".join(headlines)})
            print(f"Fetched {date}: {len(headlines)} headlines")

    if rows:
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(path, index=False)
        print(f"Saved {len(rows)} new days to {path}")
    else:
        print("No new headlines to fetch.")

    return pd.read_csv(path)

# ── 2. GROQ SENTIMENT SCORING ─────────────────────────────────────────────────

def score_sentiment(headlines_text):
    """Score a day's headlines as bullish/bearish/neutral using Groq"""
    prompt = f"""
You are a financial analyst. Given these stock market headlines, respond ONLY with a JSON object like:
{{"label": "bullish", "score": 0.7}}

Rules:
- label must be exactly one of: bullish, bearish, neutral
- score must be a float from -1.0 (most bearish) to +1.0 (most bullish)
- neutral should have a score close to 0
- No extra text, no markdown, just the JSON object

Headlines: {headlines_text}
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        print(f"Groq error: {e}")
        return {"label": "neutral", "score": 0.0}

def score_and_cache(headlines_path="data/headlines.csv",
                    scored_path="data/headlines_scored.csv"):
    """Score all unscored headlines, cache results"""
    if not os.path.exists(headlines_path):
        print("No headlines file found. Run fetch_and_cache_headlines() first.")
        return None

    df = pd.read_csv(headlines_path)

    # Load existing scored cache
    if os.path.exists(scored_path):
        scored = pd.read_csv(scored_path)
        scored_dates = set(scored["date"].astype(str).tolist())
    else:
        scored = pd.DataFrame()
        scored_dates = set()

    rows = []
    for _, row in df.iterrows():
        date = str(row["date"])
        if date in scored_dates:
            continue
        result = score_sentiment(row["headlines"])
        rows.append({
            "date": date,
            "headlines": row["headlines"],
            "sentiment_label": result["label"],
            "sentiment_score": result["score"]
        })
        print(f"Scored {date}: {result['label']} ({result['score']})")

    if rows:
        new_scored = pd.DataFrame(rows)
        combined = pd.concat([scored, new_scored], ignore_index=True)
        combined.to_csv(scored_path, index=False)
        print(f"Saved scored headlines to {scored_path}")
    else:
        print("All headlines already scored.")

    return pd.read_csv(scored_path)

# ── 3. GET TODAY'S SENTIMENT ──────────────────────────────────────────────────

def get_today_sentiment():
    """Fetch + score today's headlines, return label and score"""
    today = datetime.today().strftime("%Y-%m-%d")
    headlines = fetch_headlines(today)

    if not headlines:
        return {"label": "neutral", "score": 0.0, "headlines": []}

    text = " | ".join(headlines)
    result = score_sentiment(text)
    result["headlines"] = headlines
    return result


if __name__ == "__main__":
    print("Fetching headlines...")
    fetch_and_cache_headlines()
    print("\nScoring sentiment...")
    score_and_cache()
    print("\nToday's sentiment:")
    print(get_today_sentiment())