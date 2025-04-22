# src/data/test_fetch_stocks.py

from fetch_stocks import fetch_historical_prices, fetch_company_info

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]

    for ticker in tickers:
        print(f"\n=== {ticker} Company Info ===")
        info = fetch_company_info(ticker)
        for k, v in info.items():
            print(f"{k}: {v}")

        print(f"\n=== {ticker} Historical Prices ===")
        df = fetch_historical_prices(ticker)
        if df is not None:
            print(df.head())
