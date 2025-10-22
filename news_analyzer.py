import os
import re
import time
import json
import feedparser
import html
import logging
import argparse
from datetime import datetime, timedelta, timezone
import requests
import google.generativeai as genai

# --- Script Dependencies ---
# To run this script, you need to install the following libraries:
# pip install google-generativeai feedparser requests

# --- Configuration ---
class Config:
    """Holds all script configurations for easy access and modification."""
    GEMINI_MODEL_NAME = "gemini-2.5-flash"
    API_CALL_DELAY_SECONDS = 5
    NEWS_FETCH_DELAY_SECONDS = 0.2
    MAX_NEWS_ARTICLES_PER_TICKER = 5
    BATCH_SIZE = 20
    DEFAULT_INPUT_FILENAME = "twdata.txt"
    DEFAULT_OUTPUT_FILENAME = "news_analysis.json"
    NEWS_CACHE_FILENAME = "news_cache.json"
    NEWS_TIME_HOURS = 24
    YAHOO_FINANCE_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?count=200&formatted=true&scrIds=MOST_ACTIVES&sortField=&sortType=&start=0&useRecordsResponse=false&fields=symbol&lang=en-US&region=US"
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# --- Logging and API Initialization ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_gemini():
    """Initializes and returns the Gemini model, or None if setup fails."""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logging.warning("GEMINI_API_KEY environment variable not set. Gemini features will be disabled.")
            return None
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(Config.GEMINI_MODEL_NAME)
        logging.info("Gemini API configured successfully.")
        return model
    except ImportError:
        logging.warning("The 'google-generativeai' library is not installed. Gemini features will be disabled.")
        return None
    except Exception as e:
        logging.error(f"FATAL ERROR configuring Gemini API: {e}")
        return None

# --- Data Fetching and Parsing Functions ---
def parse_stock_tickers(filepath):
    """Parses stock tickers from a given file."""
    logging.info(f"Parsing tickers from file: {filepath}")
    tickers = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if potential_ticker := line.strip().split(' ')[0]:
                    if 1 <= len(potential_ticker) <= 5 and potential_ticker.isupper() and potential_ticker.isalpha():
                        tickers.add(potential_ticker)
    except FileNotFoundError:
        logging.warning(f"Input file '{filepath}' not found.")
    logging.info(f"Parsed {len(tickers)} unique tickers from {filepath}.")
    return tickers

def get_most_active():
    """Fetches a list of the most actively traded stock tickers."""
    logging.info("Fetching list of most active tickers from Yahoo Finance...")
    try:
        headers = {'User-Agent': Config.USER_AGENT}
        response = requests.get(Config.YAHOO_FINANCE_URL, headers=headers, timeout=10)
        response.raise_for_status()
        quotes = response.json().get('finance', {}).get('result', [{}])[0].get('quotes', [])
        tickers = {q['symbol'] for q in quotes if 'symbol' in q}
        logging.info(f"Successfully fetched {len(tickers)} active tickers.")
        return tickers
    except requests.RequestException as e:
        logging.error(f"Network error fetching most active stocks: {e}")
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error parsing Yahoo Finance API response: {e}")
    return set()

def fetch_recent_news(ticker):
    """Fetches recent news article titles for a given stock ticker."""
    urls = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={ticker}+market&hl=en-US&gl=US&ceid=US:en"
    ]
    recent_articles, seen_titles = [], set()
    time_threshold = datetime.now(timezone.utc) - timedelta(hours=Config.NEWS_TIME_HOURS)
    
    for url in urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if entry.title in seen_titles: continue
                
                published_dt = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_dt = datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc)

                if published_dt and published_dt >= time_threshold:
                    clean_title = html.unescape(entry.title)
                    recent_articles.append({'title': clean_title})
                    seen_titles.add(clean_title)
        except Exception as e:
            logging.warning(f"Could not parse news from {url} for {ticker}: {e}")
            
    return recent_articles[:Config.MAX_NEWS_ARTICLES_PER_TICKER]

# --- NEW FUNCTION: Market Sentiment Analysis ---
def get_market_sentiment_with_gemini(model, verbose=False):
    """
    Fetches news for broad market indices and uses Gemini to generate a general
    market sentiment score and summary.
    """
    logging.info("--- Starting General Market Sentiment Analysis ---")
    market_tickers = ['SPY', 'NDAQ', '^VIX']
    all_headlines = []
    
    for ticker in market_tickers:
        logging.info(f"Fetching news for market index: {ticker}")
        articles = fetch_recent_news(ticker)
        all_headlines.extend([article['title'] for article in articles])
        time.sleep(Config.NEWS_FETCH_DELAY_SECONDS)
    
    if not all_headlines:
        logging.warning("No news found for market indices. Cannot determine sentiment.")
        return

    unique_headlines = sorted(list(set(all_headlines)))
    
    prompt_parts = [
        "You are a senior market analyst. Analyze the following news headlines for key market indices (S&P 500, Nasdaq, VIX) from the last 24 hours.",
        "Synthesize this information into a general market sentiment, considering both positive and negative indicators. A high VIX is generally negative.",
        "\nYour response MUST be a single, valid JSON object with NO other text.",
        "The JSON object must have two keys:",
        "1. \"sentiment_score\": An integer from -10 (extremely bearish) to +10 (extremely bullish). Zero is neutral.",
        "2. \"summary\": A concise, one-sentence explanation for your sentiment score.",
        "\nHere are the headlines to analyze:",
        *unique_headlines
    ]
    prompt = "\n".join(prompt_parts)

    if verbose:
        logging.debug(f"--- START SENTIMENT PROMPT ---\n{prompt}\n--- END SENTIMENT PROMPT ---")

    try:
        config = genai.types.GenerationConfig(response_mime_type="application/json")
        response = model.generate_content(prompt, generation_config=config)
        sentiment_data = json.loads(response.text)
        
        score = sentiment_data.get('sentiment_score', 'N/A')
        summary = sentiment_data.get('summary', 'No summary provided.')

        logging.info("--- MARKET SENTIMENT ANALYSIS COMPLETE ---")
        logging.info(f"Overall Market Sentiment Score: {score}/10")
        logging.info(f"Summary: {summary}")
        logging.info("------------------------------------------")

    except json.JSONDecodeError:
        logging.error("Failed to decode JSON from Gemini's sentiment response.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the Gemini sentiment call: {e}")


# --- Ticker-Specific Analysis Functions ---
def analyze_news_batch_with_gemini(model, batch_data, verbose=False):
    """Analyzes a batch of tickers and their news for positive catalysts."""
    if not batch_data: return {}

    prompt_parts = [
        "You are a meticulous financial analyst. Analyze the news article titles for multiple stock tickers from the last 24 hours.",
        "For each ticker, you must perform two steps:",
        "1. **Relevance Check**: Verify the ticker is the primary subject of the news. If it's just mentioned peripherally, it does not count.",
        "2. **Catalyst Analysis**: If relevant, determine if there is a clear, positive catalyst (e.g., strong earnings, FDA approval, major contract).",
        "\nYour response MUST be a single, valid JSON object with NO other text.",
        "For each ticker key, the value must be a JSON object with three keys:",
        "1. \"has_positive_news\": A boolean (`true` or `false`).",
        "2. \"confidence\": An integer from 1 (low) to 10 (high). Must be 0 if the relevance check fails or news is not positive.",
        "3. \"summary\": A concise, one-sentence summary of the positive news. Empty string if none.",
        "\nHere is the news data:"
    ]
    for ticker, articles in batch_data.items():
        article_text = "\n".join([f"- {a['title']}" for a in articles])
        prompt_parts.append(f"\n--- TICKER: {ticker} ---\n{article_text}\n--- END TICKER: {ticker} ---")
    
    prompt = "\n".join(prompt_parts)
    if verbose:
        logging.debug(f"--- START BATCH PROMPT ---\n{prompt}\n--- END BATCH PROMPT ---")

    try:
        config = genai.types.GenerationConfig(response_mime_type="application/json")
        response = model.generate_content(prompt, generation_config=config)
        return json.loads(response.text)
    except Exception as e:
        logging.error(f"An unexpected error occurred during the Gemini batch call: {e}")
        return {
            ticker: {"has_positive_news": False, "summary": "API Error", "confidence": 0}
            for ticker in batch_data.keys()
        }

# --- Main Execution ---
def main(args):
    """Main function to orchestrate ticker fetching and analysis."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    gemini_model = initialize_gemini()
    
    # --- MODIFIED: Run market sentiment analysis first if requested ---
    if args.market_sentiment:
        if not gemini_model:
            logging.error("Exiting: Market sentiment analysis requested but Gemini model is unavailable.")
            return
        get_market_sentiment_with_gemini(gemini_model, args.verbose)
        # If ONLY sentiment was requested, we can exit here.
        if not (args.txt or args.yahoo):
            return

    # --- Continue with per-ticker analysis logic ---
    all_tickers = set()
    if args.txt:
        all_tickers.update(parse_stock_tickers(args.input))
    if args.yahoo:
        all_tickers.update(get_most_active())

    if not all_tickers:
        logging.warning("No tickers found for individual analysis. Exiting.")
        return

    sorted_tickers = sorted(list(all_tickers))
    output_data = {}
    
    try:
        with open(args.output, 'r') as f: output_data = json.load(f)
        logging.info(f"Loaded {len(output_data)} existing results from '{args.output}'.")
    except (json.JSONDecodeError, IOError, FileNotFoundError):
        logging.warning("Could not load or parse existing results file. Starting fresh.")

    if args.gemini:
        if not gemini_model:
            logging.error("Exiting: Per-ticker Gemini analysis requested but the model is unavailable.")
            return

        tickers_to_process = [t for t in sorted_tickers if t not in output_data]
        logging.info(f"Found {len(tickers_to_process)} new tickers to analyze.")

        # --- News caching logic ---
        try:
            with open(Config.NEWS_CACHE_FILENAME, 'r') as f: news_cache = json.load(f)
        except (IOError, json.JSONDecodeError): news_cache = {}

        news_data_for_analysis = {}
        for ticker in tickers_to_process:
            logging.info(f"Fetching news for {ticker}...")
            articles = fetch_recent_news(ticker)
            if articles:
                current_titles_hash = hash(frozenset(a['title'] for a in articles))
                if news_cache.get(ticker) != current_titles_hash:
                    news_data_for_analysis[ticker] = articles
                    news_cache[ticker] = current_titles_hash
                else:
                    logging.info(f"Skipping {ticker}, news has not changed since last run.")
            time.sleep(Config.NEWS_FETCH_DELAY_SECONDS)
        
        with open(Config.NEWS_CACHE_FILENAME, 'w') as f: json.dump(news_cache, f)

        tickers_with_news = list(news_data_for_analysis.keys())
        total_batches = (len(tickers_with_news) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE

        for i in range(0, len(tickers_with_news), Config.BATCH_SIZE):
            batch_tickers = tickers_with_news[i:i + Config.BATCH_SIZE]
            batch_data = {t: news_data_for_analysis[t] for t in batch_tickers}
            logging.info(f"--- Processing batch {Config.BATCH_SIZE + 1}/{total_batches} for positive catalysts ---")
            batch_results = analyze_news_batch_with_gemini(gemini_model, batch_data, args.verbose)
            
            for ticker, result in batch_results.items():
                if result.get("has_positive_news"):
                    logging.info(f"✅ Positive news for {ticker} (Confidence: {result.get('confidence')})")
                    output_data[ticker] = { "symbol": ticker, **result }
            
            if i + Config.BATCH_SIZE < len(tickers_with_news):
                 time.sleep(Config.API_CALL_DELAY_SECONDS)
    else:
        for ticker in sorted_tickers:
            if ticker not in output_data:
                output_data[ticker] = {"symbol": ticker, "has_positive_news": None, "summary": "Gemini not run."}

    try:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=4, sort_keys=True)
        logging.info(f"Successfully saved {len(output_data)} total ticker results to '{args.output}'")
    except IOError as e:
        logging.error(f"Failed to save results to file: {e}")

def setup_arg_parser():
    """Sets up and returns the argument parser."""
    parser = argparse.ArgumentParser(
        description="A tool to analyze stock news for catalysts and market sentiment using Gemini.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- MODIFIED: Added new argument ---
    parser.add_argument('--market-sentiment', action='store_true', help='Run a general market sentiment analysis using news from SPY, QQQ, and VIX.')
    
    # --- Existing arguments ---
    parser.add_argument('--txt', action='store_true', help='Parse tickers from the local input file for individual analysis.')
    parser.add_argument('--yahoo', action='store_true', help='Fetch most active tickers from Yahoo for individual analysis.')
    parser.add_argument('--gemini', action='store_true', help='Analyze individual tickers with Gemini for positive news.\nRequires GEMINI_API_KEY environment variable.')
    parser.add_argument('-i', '--input', default=Config.DEFAULT_INPUT_FILENAME, help=f"Input file path (default: {Config.DEFAULT_INPUT_FILENAME}).")
    parser.add_argument('-o', '--output', default=Config.DEFAULT_OUTPUT_FILENAME, help=f"Output JSON file path (default: {Config.DEFAULT_OUTPUT_FILENAME}).")
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging to see detailed debug info, including API prompts.')
    return parser

if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    
    import sys
    # Check if no arguments were provided for individual analysis, but sentiment was.
    has_ticker_source = '--txt' in sys.argv or '--yahoo' in sys.argv
    if len(sys.argv) == 1 or (not has_ticker_source and '--market-sentiment' not in sys.argv):
        arg_parser.print_help(sys.stderr)
        sys.exit(1)
        
    main(arg_parser.parse_args())