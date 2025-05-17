#!/usr/bin/env python3
"""
Bitcoin News Sentiment Analysis
-------------------------------
This script fetches recent Bitcoin news and analyzes sentiment using LLM.
"""

from bitcoin_utils import (
    Config, 
    fetch_bitcoin_news,
    process_news_articles,
    make_clickable,
    create_sentiment_pie_chart
)

import pandas as pd
import argparse
from datetime import datetime, timedelta
from IPython.display import HTML


def main():
    """Run the full Bitcoin news sentiment analysis pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Bitcoin News Sentiment Analysis")
    parser.add_argument("--days", type=int, default=1, 
                        help="Number of days to look back (default: 1)")
    parser.add_argument("--week", action="store_true", 
                        help="Fetch articles from the previous week")
    parser.add_argument("--from-date", type=str, 
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--to-date", type=str, 
                        help="End date in YYYY-MM-DD format")
    parser.add_argument("--page-size", type=int, default=10, 
                        help="Number of articles to fetch (default: 10)")
    args = parser.parse_args()
    
    # Calculate date range
    if args.week:
        days_back = 7
    else:
        days_back = args.days
    
    print("Bitcoin News Sentiment Analysis")
    print("===============================")
    
    # Load configuration
    config = Config.load()
    print(f"Using provider: {config.PROVIDER}")
    
    # Process articles with date range
    print("\nFetching and analyzing Bitcoin news...")
    df = process_news_articles(
        config, 
        page_size=args.page_size, 
        days_back=days_back,
        custom_from_date=args.from_date,
        custom_to_date=args.to_date
    )
    
    # Display results
    if df.empty:
        print("No articles found or analyzed.")
        return
    
    print("\nSummary Statistics:")
    print(f"Total articles analyzed: {len(df)}")
    sentiment_counts = df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count} articles ({count/len(df)*100:.1f}%)")
    
    # Save results to data folder
    config.DATA_DIR.mkdir(exist_ok=True)  # Ensure data directory exists
    output_file = config.DATA_DIR / "bitcoin_sentiment_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    print("\nDone! You can now view the results in the data folder.")


if __name__ == "__main__":
    main()
