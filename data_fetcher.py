import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import logging
import requests

class DataFetcher:
    """Handles all data fetching operations from Yahoo Finance."""
    def __init__(self, history_period="15y"):
        self.history_period = history_period
        self.user_agent = {'User-Agent': 'Mozilla/5.0'}

    def get_stock_data(self, ticker_symbol):
        """Fetches historical data for a single ticker."""
        try:
            ticker = yf.Ticker(ticker_symbol)
            price_df = ticker.history(period=self.history_period, auto_adjust=True)
            return ticker_symbol, price_df if not price_df.empty else None
        except Exception as e:
            logging.error(f"Could not fetch data for {ticker_symbol}: {e}")
            return ticker_symbol, None

    def fetch_all_data(self, tickers, max_workers=20):
        """Fetches data for a list of tickers in parallel."""
        logging.info(f"Fetching data for {len(tickers)} symbols...")
        price_data_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(self.get_stock_data, t): t for t in tickers}
            for future in future_to_ticker:
                ticker, df = future.result()
                if df is not None:
                    price_data_map[ticker] = df
        logging.info("Data fetching complete.")
        return price_data_map

    def get_most_active(self):
        """Fetches a list of the most actively traded stock tickers."""
        try:
            url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?count=200&formatted=true&scrIds=MOST_ACTIVES"
            response = requests.get(url, headers=self.user_agent, timeout=10)
            response.raise_for_status()
            quotes = response.json()['finance']['result'][0]['quotes']
            return [q['symbol'] for q in quotes]
        except Exception as e:
            logging.error(f"Could not fetch most active stocks: {e}")
            return []

