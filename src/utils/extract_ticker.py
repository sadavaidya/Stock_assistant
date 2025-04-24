company_ticker_map = {
    "apple": "AAPL",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA"
}

def get_ticker_from_query(query: str) -> str:
    query = query.lower()
    for name, ticker in company_ticker_map.items():
        if name in query:
            return ticker
    return None
