# src/data/fetch_news.py

import requests
from typing import List
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

API_KEY = os.getenv("MARKETAUX_API_KEY")

def fetch_news_from_marketaux(ticker: str, max_articles: int = 10) -> List[str]:
    """
    Fetch news headlines from Marketaux API for a given stock ticker.

    Args:
        ticker (str): Stock symbol (e.g., 'AAPL', 'TSLA')
        max_articles (int): Max number of articles to fetch

    Returns:
        List[str]: List of news headlines
    """
    if not API_KEY:
        logger.error("MARKETAUX_API_KEY not set in .env")
        return []

    url = (
        f"https://api.marketaux.com/v1/news/all?"
        f"symbols={ticker}&filter_entities=true&language=en&api_token={API_KEY}"
    )

    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
    except requests.RequestException as e:
        logger.error(f"API call failed: {e}")
        return []

    articles = data.get("data", [])[:max_articles]
    headlines = [article.get("title", "") for article in articles if article.get("title")]

    logger.info(f"Fetched {len(headlines)} headlines for {ticker}")
    return headlines
