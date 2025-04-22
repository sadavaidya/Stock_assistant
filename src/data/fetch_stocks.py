# src/data/fetch_stocks.py

import yfinance as yf
import pandas as pd
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_historical_prices(ticker: str, period: str = "30d", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetch historical stock prices using Yahoo Finance.

    Args:
        ticker (str): Stock symbol (e.g., 'AAPL')
        period (str): Time range (e.g., '30d', '1y')
        interval (str): Data frequency (e.g., '1d', '1h')

    Returns:
        pd.DataFrame: Price data with date index, or None on failure
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            logger.warning(f"No historical price data found for {ticker}")
            return None
        logger.info(f"Fetched historical prices for {ticker} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {e}")
        return None

def fetch_company_info(ticker: str) -> Dict[str, Any]:
    """
    Fetch basic company info (sector, industry, market cap, etc.).

    Args:
        ticker (str): Stock symbol

    Returns:
        dict: Company metadata
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        keys = ['longName', 'sector', 'industry', 'marketCap', 'trailingPE', 'forwardPE', 'website']
        result = {key: info.get(key, "N/A") for key in keys}
        logger.info(f"Fetched company info for {ticker}")
        return result
    except Exception as e:
        logger.error(f"Error fetching company info for {ticker}: {e}")
        return {}
