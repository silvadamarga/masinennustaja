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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import google.generativeai as genai
from dotenv import load_dotenv

# --- Load Environment Variables ---
# Loads variables from a .env file (e.g., GEMINI_API_KEY)
load_dotenv()

# --- Script Dependencies ---
# To run this script, you need to install the following libraries:
# pip install google-generativeai feedparser requests python-dotenv

# --- Configuration ---
class Config:
    """Holds all script configurations, loaded from environment variables."""
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
    API_CALL_DELAY_SECONDS = int(os.getenv("API_CALL_DELAY_SECONDS", 5))
    NEWS_FETCH_DELAY_SECONDS = float(os.getenv("NEWS_FETCH_DELAY_SECONDS", 0.2))
    MAX_NEWS_ARTICLES_PER_TICKER = int(os.getenv("MAX_NEWS_ARTICLES_PER_TICKER", 5))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 20))
    DEFAULT_INPUT_FILENAME = os.getenv("DEFAULT_INPUT_FILENAME", "twdata.txt")
    DEFAULT_OUTPUT_FILENAME = os.getenv("DEFAULT_OUTPUT_FILENAME", "news_analysis.json")
    NEWS_CACHE_FILENAME = os.getenv("NEWS_CACHE_FILENAME", "news_cache.json")
    NEWS_TIME_HOURS = int(os.getenv("NEWS_TIME_HOURS", 24))
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    YAHOO_FINANCE_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?count=200&formatted=true&scrIds=MOST_ACTIVES&sortField=&sortType=&start=0&useRecordsResponse=false&fields=symbol&lang=en-US&region=US"
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    MARKET_SENTIMENT_DB = os.getenv("MARKET_SENTIMENT_DB", "market_sentiment_db.json")

# --- Logging and Global Session ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_robust_session():
    """Creates a requests.Session with headers and automatic retries."""
    session = requests.Session()
    session.headers.update({'User-Agent': Config.USER_AGENT})
    
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504], # Retry on server errors
        backoff_factor=1  # e.g., wait 1s, 2s, 4s
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# Create a single, global session to be reused
http_session = create_robust_session()

def initialize_gemini():
    """Initializes and returns the Gemini model, or None if setup fails."""
    try:
        if not Config.GEMINI_API_KEY:
            logging.warning("GEMINI_API_KEY not set in environment or .env file. Gemini features will be disabled.")
            return None
        genai.configure(api_key=Config.GEMINI_API_KEY)
        model = genai.GenerativeModel(Config.GEMINI_MODEL_NAME)
        logging.info(f"Gemini API configured successfully with model {Config.GEMINI_MODEL_NAME}.")
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
        response = http_session.get(Config.YAHOO_FINANCE_URL, timeout=10)
        response.raise_for_status() 
        quotes = response.json().get('finance', {}).get('result', [{}])[0].get('quotes', [])
        tickers = {q['symbol'] for q in quotes if 'symbol' in q}
        logging.info(f"Successfully fetched {len(tickers)} active tickers.")
        return tickers
    except requests.RequestException as e:
        logging.error(f"Network error (retries failed) fetching most active stocks: {e}")
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error parsing Yahoo Finance API response: {e}")
    return set()

def fetch_recent_news(ticker):
    """Fetches recent news article titles and links for a given stock ticker."""
    urls = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={ticker}+market&hl=en-US&gl=US&ceid=US:en"
    ]
    recent_articles, seen_titles = [], set()
    time_threshold = datetime.now(timezone.utc) - timedelta(hours=Config.NEWS_TIME_HOURS)
    
    for url in urls:
        try:
            response = http_session.get(url, timeout=10)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            
            for entry in feed.entries:
                if entry.title in seen_titles: continue
                
                published_dt = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_dt = datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc)

                if published_dt and published_dt >= time_threshold:
                    clean_title = html.unescape(entry.title)
                    article_link = entry.get('link', '#')
                    recent_articles.append({'title': clean_title, 'link': article_link})
                    seen_titles.add(clean_title)
        except requests.RequestException as e:
            logging.warning(f"Could not fetch news from {url} for {ticker}: {e}")
        except Exception as e:
            logging.warning(f"Could not parse news from {url} for {ticker}: {e}")
            
    return recent_articles[:Config.MAX_NEWS_ARTICLES_PER_TICKER]

# --- Market Sentiment Analysis ---
def get_market_sentiment_with_gemini(model, verbose=False):
    """
    Fetches news for broad market indices and uses Gemini to generate a general
    market recommendation and summary.
    """
    logging.info("--- Starting General Market Sentiment Analysis ---")
    market_tickers = ['SPY', 'QQQ', '^VIX']
    all_headlines = []
    
    for ticker in market_tickers:
        logging.info(f"Fetching news for market index: {ticker}")
        articles = fetch_recent_news(ticker)
        all_headlines.extend([article['title'] for article in articles])
        time.sleep(Config.NEWS_FETCH_DELAY_SECONDS)
    
    if not all_headlines:
        logging.warning("No news found for market indices. Cannot determine sentiment.")
        return None

    unique_headlines = sorted(list(set(all_headlines)))
    
    prompt_parts = [
        "You are a senior market analyst. Analyze the following news headlines for key market indices (S&P 500 ETF, Nasdaq 100 ETF, VIX) from the last 24 hours.",
        "Synthesize this information into a general market sentiment, considering both positive and negative indicators. A high VIX is generally negative.",
        "Based on this sentiment, provide a market-timing recommendation for an investor looking to open *new long positions*.",
        "\nYour response MUST be a single, valid JSON object with NO other text.",
        "The JSON object must have two keys:",
        "1. \"recommendation\": A string. The value MUST be one of: \"Buy\", \"Hold\", or \"Caution\".",
        "2. \"summary\": A concise, one-sentence explanation for your recommendation.",
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
        
        recommendation = sentiment_data.get('recommendation', 'N/A')
        summary = sentiment_data.get('summary', 'No summary provided.')

        logging.info("--- MARKET SENTIMENT ANALYSIS COMPLETE ---")
        logging.info(f"Overall Market Recommendation: {recommendation}")
        logging.info(f"Summary: {summary}")
        logging.info("------------------------------------------")
        
        return sentiment_data

    except json.JSONDecodeError:
        logging.error("Failed to decode JSON from Gemini's sentiment response.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during the Gemini sentiment call: {e}")
        return None


# --- Ticker-Specific Analysis Functions ---
def analyze_news_batch_with_gemini(model, batch_data, verbose=False):
    """
    Analyzes a batch of tickers and their news for positive and negative catalysts.
    """
    if not batch_data: return {}

    prompt_parts = [
        "You are a meticulous financial analyst. Analyze the news article titles for multiple stock tickers from the last 24 hours.",
        "For each ticker, you must perform three steps:",
        "1. **Relevance Check**: Verify the ticker is the primary subject of the news. If it's just mentioned peripherally, it does not count.",
        "2. **Catalyst Analysis**: If relevant, determine if there is a clear, *positive* catalyst (e.g., strong earnings, FDA approval, major contract).",
        "3. **Negative Check**: If relevant, *also* determine if there is a clear, *negative* catalyst (e.g., earnings miss, investigation, FDA rejection).",
        "\nYour response MUST be a single, valid JSON object with NO other text.",
        "For each ticker key, the value must be a JSON object with four keys:",
        "1. \"has_positive_news\": A boolean (`true` or `false`).",
        "2. \"has_negative_news\": A boolean (`true` or `false`).",
        "3. \"confidence\": An integer from 1 (low) to 10 (high). Must be 0 if relevance fails or no clear catalyst (positive or negative) is found.",
        "4. \"summary\": A concise, one-sentence summary of the *most important* catalyst (positive or negative). Empty string if none.",
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
            ticker: {"has_positive_news": False, "has_negative_news": True, "summary": "API Error", "confidence": 0}
            for ticker in batch_data.keys()
        }

# --- Main Execution ---
def main(args):
    """Main function to orchestrate ticker fetching and analysis."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    gemini_model = initialize_gemini()
    
    # --- Market Sentiment (Append to DB) Logic ---
    if args.market_sentiment:
        if not gemini_model:
            logging.error("Exiting: Market sentiment analysis requested but Gemini model is unavailable.")
            return
        
        sentiment_result = get_market_sentiment_with_gemini(gemini_model, args.verbose)
        
        if sentiment_result:
            try:
                db_filename = Config.MARKET_SENTIMENT_DB
                try:
                    with open(db_filename, 'r') as f:
                        sentiment_db = json.load(f)
                    if not isinstance(sentiment_db, list): 
                        logging.warning(f"Sentiment DB '{db_filename}' was not a list. Resetting.")
                        sentiment_db = []
                except (IOError, json.JSONDecodeError):
                    logging.info(f"Creating new sentiment DB at '{db_filename}'.")
                    sentiment_db = [] 
                
                sentiment_result['timestamp'] = datetime.now(timezone.utc).isoformat()
                sentiment_db.append(sentiment_result)
                
                with open(db_filename, 'w') as f:
                    json.dump(sentiment_db, f, indent=4)
                
                logging.info(f"Successfully appended market sentiment to '{db_filename}'")
                
            except IOError as e:
                logging.error(f"Failed to write to market sentiment DB: {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred saving sentiment: {e}")

        if not (args.txt or args.yahoo):
            return

    # --- Per-Ticker Analysis Logic ---
    all_tickers = set()
    if args.txt:
        all_tickers.update(parse_stock_tickers(args.input))
    if args.yahoo:
        all_tickers.update(get_most_active())

    if not all_tickers:
        logging.warning("No tickers found for individual analysis. Exiting.")
        return

    # --- Timestamped output file logic ---
    base_name, ext = os.path.splitext(args.output)
    timestamp = datetime.now().strftime("%Y%m%d")
    output_filename = f"{base_name}_{timestamp}{ext}"
    logging.info(f"Per-ticker output will be saved to: {output_filename}")
    
    # --- FIXED: Load existing data from the daily file ---
    try:
        with open(output_filename, 'r') as f:
            output_data = json.load(f)
        logging.info(f"Loaded {len(output_data)} existing results from '{output_filename}'.")
    except (IOError, json.JSONDecodeError):
        logging.warning(f"Could not load existing file. Starting fresh for '{output_filename}'.")
        output_data = {} # Start fresh if file doesn't exist or is corrupt
    # --- END FIX ---

    sorted_tickers = sorted(list(all_tickers))

    if args.gemini:
        if not gemini_model:
            logging.error("Exiting: Per-ticker Gemini analysis requested but the model is unavailable.")
            return

        tickers_to_process = sorted_tickers
        logging.info(f"Checking {len(tickers_to_process)} tickers for new news...")

        # --- News caching logic ---
        try:
            with open(Config.NEWS_CACHE_FILENAME, 'r') as f: news_cache = json.load(f)
        except (IOError, json.JSONDecodeError): news_cache = {}

        news_data_for_analysis = {}
        for ticker in tickers_to_process:
            logging.info(f"Checking news for {ticker}...")
            articles = fetch_recent_news(ticker)
            if articles:
                current_titles_hash = hash(frozenset(a['title'] for a in articles))
                if news_cache.get(ticker) != current_titles_hash:
                    logging.info(f"News for {ticker} has changed. Adding to analysis queue.")
                    news_data_for_analysis[ticker] = articles
                    news_cache[ticker] = current_titles_hash
                else:
                    if ticker not in output_data:
                        logging.warning(f"Cache hit for {ticker}, but it's not in the output file. Re-adding.")
                        news_data_for_analysis[ticker] = articles # Re-queue it
                    else:
                        logging.info(f"Skipping {ticker} analysis, news has not changed.")
            time.sleep(Config.NEWS_FETCH_DELAY_SECONDS)
        
        with open(Config.NEWS_CACHE_FILENAME, 'w') as f: json.dump(news_cache, f)
        logging.info(f"News cache saved to '{Config.NEWS_CACHE_FILENAME}'.")

        tickers_with_news = list(news_data_for_analysis.keys())
        total_batches = (len(tickers_with_news) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
        if total_batches == 0:
            logging.info("No new news found for any tickers. Exiting analysis.")
            return

        for i in range(0, len(tickers_with_news), Config.BATCH_SIZE):
            batch_tickers = tickers_with_news[i:i + Config.BATCH_SIZE]
            batch_data = {t: news_data_for_analysis[t] for t in batch_tickers}
            batch_num = (i // Config.BATCH_SIZE) + 1
            logging.info(f"--- Processing batch {batch_num}/{total_batches} for catalysts ---")
            
            batch_results = analyze_news_batch_with_gemini(gemini_model, batch_data, args.verbose)
            
            for ticker, result in batch_results.items():
                is_positive = result.get("has_positive_news", False)
                is_negative = result.get("has_negative_news", False)
                
                if ticker in news_data_for_analysis:
                    result['links'] = [a['link'] for a in news_data_for_analysis[ticker]]

                if is_positive and not is_negative:
                    logging.info(f"✅ Analyzed {ticker}: Positive news.")
                elif is_negative:
                    logging.info(f"❌ AnalyZED {ticker}: Negative or Mixed news.")
                else:
                    logging.info(f"⚪️ Analyzed {ticker}: Neutral news.")
                
                # This line correctly adds or overwrites the ticker in the loaded data
                output_data[ticker] = { "symbol": ticker, **result }
            
            if i + Config.BATCH_SIZE < len(tickers_with_news):
                 time.sleep(Config.API_CALL_DELAY_SECONDS)
    else:
        logging.warning("No individual ticker analysis performed (Gemini not enabled).")

    try:
        # This saves the merged (old + new) data
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=4, sort_keys=True)
        logging.info(f"Successfully saved {len(output_data)} *total* analyzed tickers to '{output_filename}'")
    except IOError as e:
        logging.error(f"Failed to save results to file: {e}")

def setup_arg_parser():
    """Sets up and returns the argument parser."""
    parser = argparse.ArgumentParser(
        description="A tool to analyze stock news for catalysts and market sentiment using Gemini.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--market-sentiment', action='store_true', help='Run a general market sentiment analysis using news from SPY, QQQ, and VIX.')
    parser.add_argument('--txt', action='store_true', help='Parse tickers from the local input file for individual analysis.')
    parser.add.argument('--yahoo', action='store_true', help='Fetch most active tickers from Yahoo for individual analysis.')
    parser.add_argument('--gemini', action='store_true', help='Analyze individual tickers with Gemini for positive/negative news.\nRequires GEMINI_API_KEY in .env file or environment.')
    parser.add_argument('-i', '--input', default=Config.DEFAULT_INPUT_FILENAME, help=f"Input file path (default: {Config.DEFAULT_INPUT_FILENAME}).")
    parser.add_argument('-o', '--output', default=Config.DEFAULT_OUTPUT_FILENAME, help=f"Output JSON file path (default: {Config.DEFAULT_OUTPUT_FILENAME}).\nWill be timestamped, e.g., 'filename_YYYYMMDD.json'")
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging to see detailed debug info, including API prompts.')
    return parser

if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    
    import sys
    has_ticker_source = '--txt' in sys.argv or '--yahoo' in sys.argv
    if len(sys.argv) == 1 or (not has_ticker_source and '--market-sentiment' not in sys.argv):
        arg_parser.print_help(sys.stderr)
        sys.exit(1)
        
    main(arg_parser.parse_args())