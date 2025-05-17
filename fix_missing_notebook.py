import json

# Define the notebook content in JSON format
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "id": "title",
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
        {
            "cell_type": "markdown",
            "id": "setup",
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
            "id": "install",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install required packages\n",
                "!pip install -q pandas matplotlib plotly requests python-dotenv litellm tiktoken"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "config-intro",
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
            "id": "config",
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
        {
            "cell_type": "markdown",
            "id": "import-intro",
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
            "id": "imports",
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
        {
            "cell_type": "markdown",
            "id": "conclusion",
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