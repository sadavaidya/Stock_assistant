# src/data/fetch_stocks.py

import yfinance as yf
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def fetch_stock_summary_from_yfinance(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        summary = info.get("longBusinessSummary", "No company summary available.")
        current_price = info.get("currentPrice", None)
        previous_close = info.get("previousClose", None)

        if current_price is None or previous_close is None:
            raise ValueError("Price data missing")

        percent_change = ((current_price - previous_close) / previous_close) * 100
        insight = f"The stock is currently trading at ${current_price:.2f}, with a change of {percent_change:.2f}% from the previous close."

        if percent_change > 2:
            insight += " This may indicate upward momentum."
        elif percent_change < -2:
            insight += " This may indicate downward pressure."
        else:
            insight += " The price is relatively stable."

        text = f"{insight}"

        doc = {
            "text": text,
            "source": "yfinance",
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "ticker": ticker.upper()
        }

        logger.info(f"Fetched company info for {ticker}")
        return [doc]

    except Exception as e:
        logger.error(f"Error fetching stock info for {ticker}: {e}")
        return []
