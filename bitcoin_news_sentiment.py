#!/usr/bin/env python
"""Real-Time Bitcoin News Sentiment Analysis using **LiteLLM**

Fetch latest Bitcoin news, ask an LLM to score sentiment (Positive | Neutral |
Negative), aggregate the results, and append them to a CSV.

Prerequisites
-------------
1. Python 3.9+
2. pip install litellm requests python-dotenv pandas schedule
3. Two API keys stored as env vars (e.g. in a .env file):
      NEWS_API_KEY   –  https://newsapi.org/
      OPENAI_API_KEY –  your OpenAI key (or any other provider key; see docs)

Usage
-----
$ python bitcoin_news_sentiment.py          # run once
$ python bitcoin_news_sentiment.py --loop   # run once an hour
"""
from __future__ import annotations

import os
import time
import csv
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

import requests
import pandas as pd
import schedule
from dotenv import load_dotenv

from litellm import completion   # <-- LiteLLM replaces PyLLMs

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

load_dotenv()  # loads NEWS_API_KEY and OPENAI_API_KEY from a .env file if present

NEWS_API_KEY: str | None = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY: str | None = os.getenv("TOGETHER_API_KEY")
if not NEWS_API_KEY:
    raise RuntimeError("NEWS_API_KEY is not set – please export it or add to .env")

# ---- Model + endpoint configuration -----------------------------------------------------
# You can choose between different providers:
# 1. Together AI (requires API key)
# 2. Ollama (local, free)
# 3. OpenAI (requires API key)
# 4. HuggingFace (requires API key)

# Default configuration
PROVIDER = "ollama"  # Options: "together_ai", "ollama", "openai", "huggingface"
MODEL_ID = "mistral"  # For Ollama, use model names like "mistral", "llama2", etc.

# Provider-specific configurations
PROVIDER_CONFIGS = {
    "together_ai": {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "api_base": "https://api.together.xyz/v1",
        "api_key": TOGETHER_API_KEY,
    },
    "ollama": {
        "model": "mistral",
        "api_base": "http://localhost:11434",
        "api_key": None,
    },
    "openai": {
        "model": "gpt-3.5-turbo",
        "api_base": None,
        "api_key": OPENAI_API_KEY,
    },
    "huggingface": {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "api_base": "https://api-inference.huggingface.co",
        "api_key": os.getenv("HUGGINGFACE_API_KEY"),
    }
}

# Path to store CSV output
DATA_DIR = Path(__file__).with_suffix("").parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "bitcoin_news_sentiment.csv"

# ---------------------------------------------------------------------------
# Sentiment classification prompt template
# ---------------------------------------------------------------------------

PROMPT_TMPL = (
    "You are a financial sentiment analyst.\n"
    "Classify the sentiment of the following Bitcoin news article as one of "
    "'Positive', 'Neutral', or 'Negative'. Respond with ONLY the full word (not abbreviated).\n\n"
    "Article: {text}\n\nSentiment:"
)

# ---------------------------------------------------------------------------
# Step 1 – Data Ingestion
# ---------------------------------------------------------------------------

def fetch_bitcoin_news(page_size: int = 20) -> List[Dict]:
    """Fetch latest Bitcoin news articles from NewsAPI."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "Bitcoin",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("articles", [])

# ---------------------------------------------------------------------------
# Step 2 & 3 – Processing + Sentiment Analysis
# ---------------------------------------------------------------------------

def classify_sentiment(text: str) -> str:
    """Return Positive | Neutral | Negative for the supplied text."""
    prompt = PROMPT_TMPL.format(text=text.strip())
    messages = [{"role": "user", "content": prompt}]

    # Get provider configuration
    config = PROVIDER_CONFIGS[PROVIDER]
    
    # Set up API key if needed
    if config["api_key"]:
        os.environ[f"{PROVIDER.upper()}_API_KEY"] = config["api_key"]

    try:
        resp = completion(
            model=f"{PROVIDER}/{config['model']}",
            api_base=config["api_base"],
            messages=messages,
            temperature=0,
            max_tokens=1,
        )
        result = resp.choices[0].message.content.strip().split()[0]
        print(f"Debug - Model response: '{result}'")
        return result
    except Exception as e:
        print(f"Error with {PROVIDER} provider: {str(e)}")
        # Fallback to a default response if the LLM call fails
        return "Neutral"

# ---------------------------------------------------------------------------
# Step 4 – Impact Assessment helper
# ---------------------------------------------------------------------------

def sentiment_to_score(sentiment: str) -> int:
    # Handle both full and abbreviated sentiment values
    sentiment = sentiment.lower()
    print(f"Debug - Raw sentiment: '{sentiment}'")  # Debug line
    if sentiment in ["positive", "pos"]:
        return 1
    elif sentiment in ["negative", "neg"]:
        return -1
    elif sentiment in ["neutral", "ne"]:
        return 0
    print(f"Debug - No match found for sentiment: '{sentiment}'")  # Debug line
    return 0  # default fallback

# ---------------------------------------------------------------------------
# Core pipeline – pull, analyse, save
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] Fetching news …")
    articles = fetch_bitcoin_news()
    rows = []
    for art in articles:
        headline = art.get("title", "").strip()
        content = art.get("description", "") or art.get("content", "")
        if not content:
            continue
        sentiment = classify_sentiment(content)
        score = sentiment_to_score(sentiment)
        rows.append({
            "timestamp": ts,
            "publishedAt": art.get("publishedAt"),
            "headline": headline,
            "sentiment": sentiment,
            "score": score,
            "url": art.get("url"),
        })
        print(f"  → [{sentiment}] {headline[:80]}…")

    if not rows:
        print("No articles processed.")
        return

    df = pd.DataFrame(rows)
    # Append to CSV
    file_exists = CSV_PATH.exists()
    df.to_csv(CSV_PATH, mode="a", header=not file_exists, index=False)

    avg_score = df["score"].mean()
    print(f"Average sentiment score this run: {avg_score:+.2f}  ( +1 pos, 0 neu, -1 neg )")
    print(f"Saved {len(df)} rows → {CSV_PATH}")

# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------

def main(loop: bool = False):
    if loop:
        schedule.every().hour.do(run_pipeline)
        run_pipeline()  # immediate first run
        while True:
            schedule.run_pending()
            time.sleep(5)
    else:
        run_pipeline()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bitcoin News Sentiment via LiteLLM")
    parser.add_argument("--loop", action="store_true", help="Run hourly as a daemon")
    parser.add_argument("--provider", choices=list(PROVIDER_CONFIGS.keys()), 
                       default=PROVIDER, help="Choose LLM provider")
    args = parser.parse_args()
    PROVIDER = args.provider  # Override default provider if specified
    main(loop=args.loop)
