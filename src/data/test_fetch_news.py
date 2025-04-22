from fetch_news import fetch_news_from_marketaux

if __name__ == "__main__":
    headlines = fetch_news_from_marketaux("AAPL")
    for i, h in enumerate(headlines, start=1):
        print(f"{i}. {h}")
