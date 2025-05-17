import json

# Define a complete Jupyter notebook with all the needed sections
notebook = {
    "cells": [
        # Title and introduction
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Bitcoin News Sentiment Analysis Example\n",
                "\n",
                "This notebook demonstrates an end-to-end workflow for analyzing Bitcoin news sentiment, visualizing the results, and customizing the analysis for different cryptocurrencies.\n",
                "\n",
                "## Overview\n",
                "\n",
                "This example shows how to:\n",
                "1. Configure API keys and providers\n",
                "2. Fetch Bitcoin news articles\n",
                "3. Analyze sentiment using different LLM providers\n",
                "4. Visualize sentiment trends\n",
                "5. Export results to CSV\n",
                "6. Customize for different cryptocurrencies"
            ]
        },
        
        # Setup section
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Setup and Installation\n",
                "\n",
                "First, install the required packages:"
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install required packages\n",
                "%pip install -q pandas matplotlib plotly requests python-dotenv litellm tiktoken"
            ]
        },
        
        # Configuration section
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Configuration\n",
                "\n",
                "Set up environment variables and configuration. For this example, we need:\n",
                "- NEWS_API_KEY (required)\n",
                "- At least one LLM provider API key (OPENAI_API_KEY, TOGETHER_API_KEY, etc.)"
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import plotly.express as px\n",
                "import plotly.graph_objects as go\n",
                "from plotly.subplots import make_subplots\n",
                "from pathlib import Path\n",
                "from dotenv import load_dotenv\n",
                "import json\n",
                "from datetime import datetime, timedelta\n",
                "import numpy as np\n",
                "import random\n",
                "\n",
                "# Create data directory if it doesn't exist\n",
                "data_dir = Path.cwd() / \"data\"\n",
                "data_dir.mkdir(exist_ok=True)\n",
                "\n",
                "# Load environment variables\n",
                "load_dotenv()\n",
                "\n",
                "# Check if we have the required API key\n",
                "if not os.getenv(\"NEWS_API_KEY\"):\n",
                "    print(\"⚠️ WARNING: NEWS_API_KEY not found in .env file\")\n",
                "    print(\"You should create a .env file with your API keys:\")\n",
                "    print(\"NEWS_API_KEY=your_newsapi_key\")\n",
                "    print(\"OPENAI_API_KEY=your_openai_key  # optional\")\n",
                "    print(\"TOGETHER_API_KEY=your_together_key  # optional\")\n",
                "    # For demo purposes, we'll skip the API calls and work with sample data\n",
                "    DEMO_MODE = True\n",
                "else:\n",
                "    DEMO_MODE = False\n",
                "    print(\"✅ API keys loaded successfully\")"
            ]
        },
        
        # Import utilities
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Import Utilities\n",
                "\n",
                "Now let's import the utility functions from `bitcoin_utils.py`. If we're in demo mode, we'll create some sample data for demonstration."
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if not DEMO_MODE:\n",
                "    # Import our actual utility functions\n",
                "    try:\n",
                "        from bitcoin_utils import (\n",
                "            Config, fetch_bitcoin_news, classify_sentiment,\n",
                "            process_news_articles, make_clickable, create_sentiment_pie_chart\n",
                "        )\n",
                "        config = Config.load()\n",
                "        print(f\"Using LLM provider: {config.PROVIDER}\")\n",
                "    except ImportError:\n",
                "        print(\"Could not import bitcoin_utils. Using DEMO_MODE instead.\")\n",
                "        DEMO_MODE = True\n",
                "\n",
                "# For demo mode, create sample data\n",
                "if DEMO_MODE:\n",
                "    print(\"Running in DEMO_MODE with sample data\")\n",
                "    \n",
                "    # Create sample data function\n",
                "    def create_sample_data(days=30, ticker=\"Bitcoin\"):\n",
                "        \"\"\"Create sample sentiment data for demonstration\"\"\"\n",
                "        \n",
                "        # Generate dates\n",
                "        end_date = datetime.now()\n",
                "        start_date = end_date - timedelta(days=days)\n",
                "        dates = pd.date_range(start=start_date, end=end_date, freq='D')\n",
                "        \n",
                "        # Create sample headlines\n",
                "        headlines = [\n",
                "            f\"{ticker} surges to new heights as investors flock to crypto\",\n",
                "            f\"{ticker} stabilizes as market uncertainty continues\",\n",
                "            f\"Regulations may impact {ticker} growth, experts say\",\n",
                "            f\"New {ticker} ETF approval drives positive market sentiment\",\n",
                "            f\"{ticker} sees correction after recent gains\",\n",
                "            f\"Analysts predict bright future for {ticker} despite volatility\",\n",
                "            f\"{ticker} under pressure as traditional markets rebound\",\n",
                "            f\"Institutional investors increase {ticker} holdings\",\n",
                "            f\"Market fears impact {ticker} price in short term\",\n",
                "            f\"{ticker} community optimistic about upcoming protocol upgrade\"\n",
                "        ]\n",
                "        \n",
                "        # Generate random sentiment with some trends\n",
                "        sentiments = []\n",
                "        scores = []\n",
                "        trend = np.sin(np.linspace(0, 3*np.pi, len(dates))) * 0.5  # Oscillating trend\n",
                "        \n",
                "        for i, date in enumerate(dates):\n",
                "            # Add trend and randomness\n",
                "            r = random.random() + trend[i]\n",
                "            if r > 1.0:\n",
                "                sentiment = \"Positive\"\n",
                "                score = 1\n",
                "            elif r < 0.0:\n",
                "                sentiment = \"Negative\"\n",
                "                score = -1\n",
                "            else:\n",
                "                sentiment = \"Neutral\"\n",
                "                score = 0\n",
                "            \n",
                "            sentiments.append(sentiment)\n",
                "            scores.append(score)\n",
                "        \n",
                "        # Create DataFrame\n",
                "        data = []\n",
                "        for i, date in enumerate(dates):\n",
                "            # Create multiple entries per day\n",
                "            for j in range(random.randint(1, 3)):\n",
                "                headline = random.choice(headlines)\n",
                "                # Add slight variation to sentiment based on headline\n",
                "                s_score = scores[i]\n",
                "                if \"surges\" in headline or \"bright future\" in headline or \"optimistic\" in headline:\n",
                "                    s_score = min(1, s_score + 0.5)\n",
                "                elif \"pressure\" in headline or \"fears\" in headline or \"uncertainty\" in headline:\n",
                "                    s_score = max(-1, s_score - 0.5)\n",
                "                \n",
                "                if s_score >= 0.5:\n",
                "                    sentiment = \"Positive\"\n",
                "                    score = 1\n",
                "                elif s_score <= -0.5:\n",
                "                    sentiment = \"Negative\"\n",
                "                    score = -1\n",
                "                else:\n",
                "                    sentiment = \"Neutral\"\n",
                "                    score = 0\n",
                "                \n",
                "                data.append({\n",
                "                    'publishedAt': date + timedelta(hours=random.randint(0, 23)),\n",
                "                    'headline': headline,\n",
                "                    'sentiment': sentiment,\n",
                "                    'score': score,\n",
                "                    'url': f\"https://example.com/news/{i}-{j}\",\n",
                "                    'cached': False\n",
                "                })\n",
                "        \n",
                "        return pd.DataFrame(data)\n",
                "    \n",
                "    # Create sample DataFrame\n",
                "    sample_df = create_sample_data(days=30)\n",
                "    \n",
                "    # Save to CSV\n",
                "    csv_path = data_dir / \"bitcoin_sentiment_results.csv\"\n",
                "    sample_df.to_csv(csv_path, index=False)\n",
                "    print(f\"Sample data saved to {csv_path}\")"
            ]
        },
        
        # Fetch and analyze news
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Fetch and Analyze News\n",
                "\n",
                "Now let's fetch Bitcoin news and analyze the sentiment. If we're in demo mode, we'll use the sample data created above."
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if not DEMO_MODE:\n",
                "    # Process Bitcoin news articles\n",
                "    print(\"Fetching and analyzing Bitcoin news...\")\n",
                "    df = process_news_articles(config, page_size=10, days_back=7)\n",
                "    \n",
                "    # Save to CSV\n",
                "    csv_path = data_dir / \"bitcoin_sentiment_results.csv\"\n",
                "    df.to_csv(csv_path, index=False)\n",
                "    print(f\"Results saved to {csv_path}\")\n",
                "else:\n",
                "    print(\"Skipping API calls (DEMO_MODE)\")\n",
                "    # Use the sample data we created earlier\n",
                "    csv_path = data_dir / \"bitcoin_sentiment_results.csv\"\n",
                "    df = pd.read_csv(csv_path)\n",
                "    df['publishedAt'] = pd.to_datetime(df['publishedAt'])"
            ]
        },
        
        # Data exploration
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Data Exploration and Visualization\n",
                "\n",
                "Now let's explore the data and create visualizations to understand the sentiment distribution."
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Display basic info\n",
                "print(f\"Loaded {len(df)} articles from {df['publishedAt'].min().date()} to {df['publishedAt'].max().date()}\")\n",
                "print(\"\\nSentiment distribution:\")\n",
                "sentiment_counts = df['sentiment'].value_counts()\n",
                "print(sentiment_counts)\n",
                "\n",
                "# Calculate percentage of positive vs negative\n",
                "total = len(df)\n",
                "positive_pct = sentiment_counts.get('Positive', 0) / total * 100\n",
                "negative_pct = sentiment_counts.get('Negative', 0) / total * 100\n",
                "neutral_pct = sentiment_counts.get('Neutral', 0) / total * 100\n",
                "\n",
                "print(f\"\\nPositive: {positive_pct:.1f}%\")\n",
                "print(f\"Neutral: {neutral_pct:.1f}%\")\n",
                "print(f\"Negative: {negative_pct:.1f}%\")\n",
                "\n",
                "# Show sample of the data\n",
                "df.head()"
            ]
        },
        
        # Pie chart
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.1 Pie Chart Visualization\n",
                "\n",
                "Let's create a pie chart to visualize the sentiment distribution."
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create a pie chart\n",
                "labels = sentiment_counts.index\n",
                "values = sentiment_counts.values\n",
                "colors = ['green' if x == 'Positive' else 'red' if x == 'Negative' else 'gray' for x in labels]\n",
                "\n",
                "fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker_colors=colors)])\n",
                "fig.update_layout(title_text=\"Bitcoin News Sentiment Distribution\")\n",
                "fig.show()"
            ]
        },
        
        # Time series
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.2 Time Series Visualization\n",
                "\n",
                "Let's create a time series visualization to see how sentiment has changed over time."
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Group by date and calculate sentiment percentages\n",
                "df['date'] = df['publishedAt'].dt.date\n",
                "sentiment_by_date = df.groupby('date')['sentiment'].value_counts().unstack().fillna(0)\n",
                "\n",
                "# Calculate percentages\n",
                "for col in sentiment_by_date.columns:\n",
                "    sentiment_by_date[f'{col}_pct'] = sentiment_by_date[col] / sentiment_by_date.sum(axis=1) * 100\n",
                "\n",
                "# Create figure\n",
                "plt.figure(figsize=(12, 6))\n",
                "\n",
                "# Plot percentage of positive and negative sentiment\n",
                "if 'Positive' in sentiment_by_date.columns:\n",
                "    plt.plot(sentiment_by_date.index, sentiment_by_date['Positive_pct'], \n",
                "             color='green', marker='o', linestyle='-', label='Positive %')\n",
                "if 'Negative' in sentiment_by_date.columns:\n",
                "    plt.plot(sentiment_by_date.index, sentiment_by_date['Negative_pct'], \n",
                "             color='red', marker='x', linestyle='-', label='Negative %')\n",
                "\n",
                "plt.title('Bitcoin News Sentiment Over Time (% Positive vs Negative)')\n",
                "plt.xlabel('Date')\n",
                "plt.ylabel('Percentage')\n",
                "plt.legend()\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        
        # Rolling average
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.3 Rolling Average Sentiment Score\n",
                "\n",
                "Let's calculate and visualize a 7-day rolling average of the sentiment score."
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Calculate rolling sentiment score (7-day window)\n",
                "daily_score = df.groupby('date')['score'].mean()\n",
                "rolling_score = daily_score.rolling(window=7, min_periods=1).mean()\n",
                "\n",
                "plt.figure(figsize=(12, 6))\n",
                "plt.plot(rolling_score.index, rolling_score, color='blue', linewidth=2)\n",
                "plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)\n",
                "plt.title('7-Day Rolling Average Sentiment Score')\n",
                "plt.xlabel('Date')\n",
                "plt.ylabel('Average Sentiment Score (-1 to 1)')\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        
        # Interactive visualization
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.4 Interactive Visualization with Plotly\n",
                "\n",
                "Now let's create an interactive visualization using Plotly for better exploration."
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create an interactive time series\n",
                "fig = make_subplots(rows=2, cols=1, \n",
                "                   subplot_titles=('Sentiment Distribution Over Time', 'Rolling Average Sentiment Score'))\n",
                "\n",
                "# Add sentiment distribution\n",
                "for sentiment, color in zip(['Positive', 'Neutral', 'Negative'], ['green', 'gray', 'red']):\n",
                "    if sentiment in sentiment_by_date.columns:\n",
                "        fig.add_trace(\n",
                "            go.Scatter(\n",
                "                x=sentiment_by_date.index, \n",
                "                y=sentiment_by_date[f'{sentiment}_pct'],\n",
                "                mode='lines+markers',\n",
                "                name=f'{sentiment} %',\n",
                "                line=dict(color=color),\n",
                "                hovertemplate='%{y:.1f}%<extra></extra>'\n",
                "            ),\n",
                "            row=1, col=1\n",
                "        )\n",
                "\n",
                "# Add rolling sentiment score\n",
                "fig.add_trace(\n",
                "    go.Scatter(\n",
                "        x=rolling_score.index,\n",
                "        y=rolling_score,\n",
                "        mode='lines',\n",
                "        name='7-Day Avg Score',\n",
                "        line=dict(color='royalblue', width=3),\n",
                "        hovertemplate='Score: %{y:.2f}<extra></extra>'\n",
                "    ),\n",
                "    row=2, col=1\n",
                ")\n",
                "\n",
                "# Add horizontal line at y=0 for the second subplot\n",
                "fig.add_shape(\n",
                "    type='line',\n",
                "    x0=rolling_score.index.min(),\n",
                "    x1=rolling_score.index.max(),\n",
                "    y0=0, y1=0,\n",
                "    line=dict(color='gray', dash='dash'),\n",
                "    row=2, col=1\n",
                ")\n",
                "\n",
                "# Update layout\n",
                "fig.update_layout(\n",
                "    title='Bitcoin News Sentiment Analysis',\n",
                "    height=800,\n",
                "    hovermode='x unified',\n",
                "    showlegend=True,\n",
                ")\n",
                "\n",
                "fig.update_yaxes(title_text='Percentage', row=1, col=1)\n",
                "fig.update_yaxes(title_text='Sentiment Score', row=2, col=1)\n",
                "\n",
                "fig.show()"
            ]
        },
        
        # Headlines by sentiment
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Examining Headlines by Sentiment\n",
                "\n",
                "Let's look at examples of headlines for each sentiment category."
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Display examples of headlines by sentiment\n",
                "for sentiment in ['Positive', 'Neutral', 'Negative']:\n",
                "    headlines = df[df['sentiment'] == sentiment]['headline'].unique()[:3]\n",
                "    print(f\"\\n{sentiment} Headlines:\")\n",
                "    for headline in headlines:\n",
                "        print(f\"  • {headline}\")"
            ]
        },
        
        # Customizing for different cryptocurrencies
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Customizing for Different Cryptocurrencies\n",
                "\n",
                "One of the strengths of this system is how easy it is to adapt for different cryptocurrencies. Let's demonstrate how to customize it for Ethereum."
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def analyze_crypto_sentiment(crypto=\"Bitcoin\", days_back=7, page_size=10, demo=DEMO_MODE):\n",
                "    \"\"\"Analyze news sentiment for different cryptocurrencies\"\"\"\n",
                "    if not demo:\n",
                "        # In a real implementation, you would modify the search query in fetch_bitcoin_news\n",
                "        # Here's a simplified example (would need to be adjusted for the actual implementation)\n",
                "        try:\n",
                "            from bitcoin_utils import Config, process_news_articles\n",
                "            \n",
                "            config = Config.load()\n",
                "            \n",
                "            # This is a placeholder for how you might modify the function in a real implementation\n",
                "            # In a complete implementation, you would modify the fetch_bitcoin_news function\n",
                "            # to accept a query parameter or modify the process_news_articles function\n",
                "            print(f\"Fetching and analyzing {crypto} news...\")\n",
                "            print(f\"In a real implementation, you would modify the API query from 'Bitcoin' to '{crypto}'\")\n",
                "            \n",
                "            # Placeholder for actual API call\n",
                "            print(f\"This would call: process_news_articles(config, page_size={page_size}, days_back={days_back}, query='{crypto}')\")\n",
                "            \n",
                "        except ImportError:\n",
                "            print(\"Could not import bitcoin_utils. Using demo data instead.\")\n",
                "            demo = True\n",
                "    \n",
                "    if demo:\n",
                "        # Create sample data for the specified crypto\n",
                "        print(f\"Creating sample data for {crypto}\")\n",
                "        sample_df = create_sample_data(days=days_back, ticker=crypto)\n",
                "        \n",
                "        # Save to CSV with crypto name\n",
                "        crypto_lower = crypto.lower()\n",
                "        csv_path = data_dir / f\"{crypto_lower}_sentiment_results.csv\"\n",
                "        sample_df.to_csv(csv_path, index=False)\n",
                "        print(f\"Sample {crypto} data saved to {csv_path}\")\n",
                "        \n",
                "        return sample_df\n",
                "    \n",
                "    return None  # In a real implementation, this would return the analyzed data\n",
                "\n",
                "# Demo for Ethereum\n",
                "ethereum_df = analyze_crypto_sentiment(\"Ethereum\", days_back=14)\n",
                "\n",
                "if ethereum_df is not None:\n",
                "    # Quick visualization of Ethereum sentiment\n",
                "    eth_sentiment_counts = ethereum_df['sentiment'].value_counts()\n",
                "    \n",
                "    # Create pie chart\n",
                "    labels = eth_sentiment_counts.index\n",
                "    values = eth_sentiment_counts.values\n",
                "    colors = ['green' if x == 'Positive' else 'red' if x == 'Negative' else 'gray' for x in labels]\n",
                "    \n",
                "    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker_colors=colors)])\n",
                "    fig.update_layout(title_text=f\"Ethereum News Sentiment Distribution\")\n",
                "    fig.show()\n",
                "    \n",
                "    # Show example headlines\n",
                "    print(\"\\nSample Ethereum Headlines:\")\n",
                "    for sentiment in ['Positive', 'Neutral', 'Negative']:\n",
                "        headlines = ethereum_df[ethereum_df['sentiment'] == sentiment]['headline'].unique()[:2]\n",
                "        print(f\"\\n{sentiment} Headlines:\")\n",
                "        for headline in headlines:\n",
                "            print(f\"  • {headline}\")"
            ]
        },
        
        # Conclusion
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Conclusion\n",
                "\n",
                "This notebook demonstrated a complete workflow for Bitcoin news sentiment analysis:\n",
                "\n",
                "1. Setting up the environment and configuration\n",
                "2. Fetching and analyzing news articles\n",
                "3. Loading results from CSV\n",
                "4. Creating visualizations of sentiment over time\n",
                "5. Customizing the analysis for different cryptocurrencies\n",
                "\n",
                "The modular design of the system makes it easy to:\n",
                "- Schedule regular sentiment analysis updates\n",
                "- Customize for different crypto assets\n",
                "- Visualize the results in various formats\n",
                "- Integrate with trading strategies or dashboards\n",
                "\n",
                "For production use, you would typically set up this system to run on a schedule and feed into a dashboard or trading algorithm."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Write the notebook to a file
with open('bitcoin.example.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Example notebook 'bitcoin.example.ipynb' created successfully.") 