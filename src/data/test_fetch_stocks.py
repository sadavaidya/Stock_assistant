# src/data/test_fetch_stocks.py

from fetch_stocks import fetch_stock_summary_from_yfinance

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]

    for ticker in tickers:
        from pprint import pprint
        pprint(fetch_stock_summary_from_yfinance(ticker))

