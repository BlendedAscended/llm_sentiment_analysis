#!/usr/bin/env python
"""Real-Time Bitcoin News Sentiment Analysis using **LiteLLM**

Fetch latest Bitcoin news, ask an LLM to score sentiment (Positive | Neutral |
Negative), aggregate the results, and append them to a CSV.

Prerequisites
-------------
1. Python 3.9+
2. pip install litellm requests python-dotenv pandas schedule tiktoken
3. Two API keys stored as env vars (e.g. in a .env file):
      NEWS_API_KEY   –  https://newsapi.org/
      OPENAI_API_KEY –  your OpenAI key (or any other provider key; see docs)
      GOOGLE_API_KEY –  your Google API key for Gemini

Usage
-----
$ python bitcoin_news_sentiment.py          # run once
$ python bitcoin_news_sentiment.py --loop   # run once an hour
"""
from __future__ import annotations

import os
import time
import csv
import json
import hashlib
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

import requests
import pandas as pd
import schedule
from dotenv import load_dotenv
import tiktoken
from litellm import completion

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Configuration settings for the application."""
    NEWS_API_KEY: str
    OPENAI_API_KEY: Optional[str]
    TOGETHER_API_KEY: Optional[str]
    GOOGLE_API_KEY: Optional[str]
    HUGGINGFACE_API_KEY: Optional[str]
    PROVIDER: str = "together_ai"
    MODEL_ID: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    MAX_TOKENS: int = 2000
    DATA_DIR: Path = Path(__file__).with_suffix("").parent / "data"
    CSV_PATH: Path = DATA_DIR / "bitcoin_news_sentiment.csv"
    CACHE_PATH: Path = DATA_DIR / "article_cache.json"
    PROMPT_TEMPLATE: str = (
        "You are a financial sentiment analyst.\n"
        "Classify the sentiment of the following Bitcoin news article as one of "
        "'Positive', 'Neutral', or 'Negative'. Respond with ONLY the full word (not abbreviated).\n\n"
        "Article: {text}\n\nSentiment:"
    )

    @classmethod
    def load(cls) -> 'Config':
        """Load configuration from environment variables."""
        load_dotenv()
        
        if not os.getenv("NEWS_API_KEY"):
            raise RuntimeError("NEWS_API_KEY is not set – please export it or add to .env")
            
        return cls(
            NEWS_API_KEY=os.getenv("NEWS_API_KEY", ""),
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
            TOGETHER_API_KEY=os.getenv("TOGETHER_API_KEY"),
            GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY"),
            HUGGINGFACE_API_KEY=os.getenv("HUGGINGFACE_API_KEY"),
        )

    @property
    def provider_configs(self) -> Dict[str, Dict]:
        """Get provider-specific configurations."""
        return {
            "together_ai": {
                "model": "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
                "api_key": self.TOGETHER_API_KEY,
            },
            "ollama": {
                "model": "ollama/llama2",
            },
            "openai": {
                "model": "gpt-3.5-turbo",
                "api_key": self.OPENAI_API_KEY,
            },
            "huggingface": {
                "model": "huggingface/HuggingFaceH4/zephyr-7b-beta",
                "api_key": self.HUGGINGFACE_API_KEY,
            },
            "gemini": {
                "model": "gemini-pro",
                "api_key": self.GOOGLE_API_KEY,
            }
        }

# ---------------------------------------------------------------------------
# Base Classes
# ---------------------------------------------------------------------------

class NewsSource(ABC):
    """Abstract base class for news sources."""
    
    @abstractmethod
    def fetch_news(self, page_size: int = 20) -> List[Dict]:
        """Fetch news articles from the source."""
        pass

class SentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers."""
    
    @abstractmethod
    def analyze(self, text: str) -> Tuple[str, float]:
        """Analyze the sentiment of the given text."""
        pass

class DataStore(ABC):
    """Abstract base class for data storage."""
    
    @abstractmethod
    def save(self, data: List[Dict]) -> None:
        """Save the data."""
        pass
    
    @abstractmethod
    def load(self) -> List[Dict]:
        """Load the data."""
        pass

# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

class NewsAPISource(NewsSource):
    """NewsAPI implementation of NewsSource."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def fetch_news(self, page_size: int = 20) -> List[Dict]:
        """Fetch latest Bitcoin news articles from NewsAPI."""
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "Bitcoin",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": self.api_key,
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("articles", [])

class LiteLLMAnalyzer(SentimentAnalyzer):
    """LiteLLM implementation of SentimentAnalyzer."""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))
    
    def truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limit, preserving complete sentences."""
        if self.count_tokens(text) <= self.config.MAX_TOKENS:
            return text
        
        sentences = text.split('. ')
        truncated = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            if current_tokens + sentence_tokens <= self.config.MAX_TOKENS:
                truncated.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        return '. '.join(truncated) + '.'
    
    def analyze(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment using LiteLLM."""
        truncated_text = self.truncate_text(text)
        prompt = self.config.PROMPT_TEMPLATE.format(text=truncated_text.strip())
        
        try:
            config = self.config.provider_configs[self.config.PROVIDER]
            
            response = completion(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1,
                api_key=config.get("api_key"),
            )
            
            result = response.choices[0].message.content.strip().split()[0]
            print(f"Debug - Model response: '{result}'")
            
            return result, 0.0  # Cost is handled by LiteLLM
        except Exception as e:
            print(f"Error with {self.config.PROVIDER} provider: {str(e)}")
            return "Neutral", 0.0

class CSVDataStore(DataStore):
    """CSV implementation of DataStore."""
    
    def __init__(self, config: Config):
        self.config = config
        self.config.DATA_DIR.mkdir(exist_ok=True)
    
    def save(self, data: List[Dict]) -> None:
        """Save data to CSV."""
        df = pd.DataFrame(data)
        file_exists = self.config.CSV_PATH.exists()
        df.to_csv(self.config.CSV_PATH, mode="a", header=not file_exists, index=False)
    
    def load(self) -> List[Dict]:
        """Load data from CSV."""
        if not self.config.CSV_PATH.exists():
            return []
        return pd.read_csv(self.config.CSV_PATH).to_dict('records')

class Cache:
    """Article cache implementation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.cache: Dict[str, Dict] = {}
        self.load()
    
    def load(self) -> None:
        """Load cache from disk."""
        if self.config.CACHE_PATH.exists():
            with open(self.config.CACHE_PATH, 'r') as f:
                self.cache = json.load(f)
    
    def save(self) -> None:
        """Save cache to disk."""
        with open(self.config.CACHE_PATH, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def get_article_hash(self, article: Dict) -> str:
        """Generate a unique hash for an article."""
        content = f"{article.get('title', '')}{article.get('description', '')}{article.get('url', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_cached(self, article: Dict) -> bool:
        """Check if an article is in the cache."""
        return self.get_article_hash(article) in self.cache
    
    def add(self, article: Dict, sentiment: str, score: int, cost: float) -> None:
        """Add an article to the cache."""
        article_hash = self.get_article_hash(article)
        self.cache[article_hash] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "headline": article.get("title", "").strip(),
            "sentiment": sentiment,
            "score": score,
            "url": article.get("url"),
            "cost": cost,
        }

class SentimentPipeline:
    """Main pipeline for sentiment analysis."""
    
    def __init__(self, config: Config):
        self.config = config
        self.news_source = NewsAPISource(config.NEWS_API_KEY)
        self.analyzer = LiteLLMAnalyzer(config)
        self.data_store = CSVDataStore(config)
        self.cache = Cache(config)
    
    def sentiment_to_score(self, sentiment: str) -> int:
        """Convert sentiment to numerical score."""
        sentiment = sentiment.lower()
        if sentiment in ["positive", "pos"]:
            return 1
        elif sentiment in ["negative", "neg"]:
            return -1
        elif sentiment in ["neutral", "ne"]:
            return 0
        return 0
    
    def run(self) -> None:
        """Run the sentiment analysis pipeline."""
        ts = datetime.now(timezone.utc).isoformat()
        print(f"[{ts}] Fetching news …")
        
        articles = self.news_source.fetch_news()
        rows = []
        total_cost = 0.0
        
        for art in articles:
            if self.cache.is_cached(art):
                print(f"  → Skipping cached article: {art.get('title', '')[:80]}…")
                continue
            
            headline = art.get("title", "").strip()
            content = art.get("description", "") or art.get("content", "")
            if not content:
                continue
            
            sentiment, cost = self.analyzer.analyze(content)
            total_cost += cost
            score = self.sentiment_to_score(sentiment)
            
            self.cache.add(art, sentiment, score, cost)
            
            rows.append({
                "timestamp": ts,
                "publishedAt": art.get("publishedAt"),
                "headline": headline,
                "sentiment": sentiment,
                "score": score,
                "url": art.get("url"),
                "cost": cost,
            })
            print(f"  → [{sentiment}] {headline[:80]}… (Cost: ${cost:.6f})")
        
        if not rows:
            print("No new articles processed.")
            return
        
        self.cache.save()
        self.data_store.save(rows)
        
        df = pd.DataFrame(rows)
        avg_score = df["score"].mean()
        print(f"Average sentiment score this run: {avg_score:+.2f}  ( +1 pos, 0 neu, -1 neg )")
        print(f"Total cost this run: ${total_cost:.6f}")
        print(f"Saved {len(df)} rows → {self.config.CSV_PATH}")
        
        print("\nCost Summary:")
        print(f"Provider: {self.config.PROVIDER}")
        print(f"Model: {self.config.provider_configs[self.config.PROVIDER]['model']}")
        print(f"Total cost: ${total_cost:.6f}")
        print(f"Average cost per article: ${total_cost/len(rows):.6f}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(loop: bool = False, provider: str = "together_ai"):
    """Main entry point."""
    config = Config.load()
    config.PROVIDER = provider
    pipeline = SentimentPipeline(config)
    
    if loop:
        schedule.every().hour.do(pipeline.run)
        pipeline.run()  # immediate first run
        while True:
            schedule.run_pending()
            time.sleep(5)
    else:
        pipeline.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bitcoin News Sentiment via LiteLLM")
    parser.add_argument("--loop", action="store_true", help="Run hourly as a daemon")
    parser.add_argument("--provider", choices=["together_ai", "ollama", "openai", "huggingface", "gemini"],
                       default="together_ai", help="Choose LLM provider")
    args = parser.parse_args()
    main(loop=args.loop, provider=args.provider)
