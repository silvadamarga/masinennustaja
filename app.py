import os
import json
import logging
import argparse
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from concurrent.futures import ThreadPoolExecutor

# --- Local Imports ---
from prediction_service import PredictionService
from data_fetcher import DataFetcher

# --- GPU Configuration Function ---
def setup_gpu_memory_growth():
    """Checks for GPUs and enables memory growth to prevent VRAM hogging."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"TensorFlow memory growth enabled for {len(gpus)} GPU(s).")
        else:
            logging.info("No GPU found, running on CPU.")
    except RuntimeError as e:
        logging.error(f"Error setting memory growth: {e}")

# --- Initialize Flask App ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Instantiate Services ---
prediction_service = None
data_fetcher = None

# --- Configuration ---
TICKER_SOURCE_FILE = "news_analysis.json"

# --- Helper Functions ---
def load_news_data(filepath=TICKER_SOURCE_FILE):
    """Loads news analysis data from the JSON file."""
    if not os.path.exists(filepath):
        logging.warning(f"News analysis file not found: {filepath}")
        return {}
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading or parsing {filepath}: {e}")
        return {}

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main user interface."""
    model_load_error = "Critical Error: Model artifacts failed to load. The application is not functional." if not prediction_service else None
    return render_template('index.html', error=model_load_error)

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Handles the asynchronous analysis request from the frontend.
    The regime filter logic is now handled on the client side.
    """
    if not prediction_service or not data_fetcher:
        return jsonify({"error": "A backend service is not available. Please check server logs."}), 500

    data = request.get_json()
    tickers = data.get('tickers', [])
    if not tickers:
        return jsonify({"error": "No tickers provided."}), 400

    # --- Step 1: Fetch all required data for both models ---
    # The new V4 model requires more tickers for its features
    regime_tickers = ["SPY", "^VIX", "^TNX", "^FVX", "HYG", "IEF", "GLD"]
    all_tickers_to_fetch = list(set(tickers + regime_tickers))
    price_data_map = data_fetcher.fetch_all_data(all_tickers_to_fetch)
    
    # --- Step 2: Predict the CURRENT Market Regime ---
    regime_result = prediction_service.predict_current_regime(price_data_map)
    predicted_regime = regime_result.get('regime_id') if isinstance(regime_result, dict) else None

    # --- Step 3: Prepare data for breakout prediction ---
    spy_df = price_data_map.get("SPY")
    if spy_df is None:
        return jsonify({"error": "Could not fetch SPY data. Analysis cannot proceed."}), 500
    
    # Create a clean map for breakout prediction containing only user-requested tickers
    breakout_price_map = {t: df for t, df in price_data_map.items() if t in tickers}

    prepared_data_list = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(prediction_service.prepare_data_for_prediction, ticker, df, spy_df, predicted_regime): ticker
            for ticker, df in breakout_price_map.items()
        }
        for future in future_to_ticker:
            try:
                prepared_data_list.append(future.result())
            except Exception as e:
                ticker = future_to_ticker[future]
                logging.error(f"Data preparation for {ticker} failed in thread: {e}", exc_info=True)
                prepared_data_list.append({"Ticker": ticker, "Signal_Raw": "Prep Failed", "Signal_Filtered": "Prep Failed"})
    
    # --- Step 4: Run batch prediction. The service now returns both raw and filtered signals. ---
    technical_results = prediction_service.run_batch_prediction(prepared_data_list, predicted_regime)
    
    # --- Step 5: Merge with News Data ---
    news_data = load_news_data()
    final_results = []
    for result in technical_results:
        ticker = result['Ticker']
        if ticker_news := news_data.get(ticker):
            result['News_Confidence'] = f"{ticker_news.get('confidence', 0)}/10"
            result['News_Summary'] = ticker_news.get('summary', 'N/A')
        else:
            result['News_Confidence'] = 'N/A'
            result['News_Summary'] = 'N/A'
        final_results.append(result)
        
    return jsonify({
        "results": sorted(final_results, key=lambda x: x.get('Ticker', '')),
        "regime": regime_result
    })

@app.route('/get-tickers-from-file')
def get_tickers_from_file():
    if not os.path.exists(TICKER_SOURCE_FILE):
        return jsonify({"error": f"File not found: {TICKER_SOURCE_FILE}"}), 404
    try:
        with open(TICKER_SOURCE_FILE, 'r') as f: data = json.load(f)
        return jsonify({"tickers": list(data.keys())})
    except Exception as e:
        logging.error(f"Error reading or parsing {TICKER_SOURCE_FILE}: {e}")
        return jsonify({"error": f"Failed to process {TICKER_SOURCE_FILE}."}), 500

@app.route('/get-yahoo-most-active')
def get_yahoo_most_active():
    if not data_fetcher:
        return jsonify({"error": "Data fetching service is not available."}), 500
    try:
        tickers = data_fetcher.get_most_active()
        return jsonify({"tickers": tickers}) if tickers else jsonify({"error": "Failed to fetch most active tickers."}), 500
    except Exception as e:
        logging.error(f"Error in get_yahoo_most_active route: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Stock Breakout Predictor Flask App")
    parser.add_argument('--gpu-mem-growth', action='store_true', help="Enable TensorFlow GPU memory growth.")
    return parser

if __name__ == '__main__':
    parser = setup_arg_parser()
    args = parser.parse_args()

    if args.gpu_mem_growth:
        setup_gpu_memory_growth()

    try:
        prediction_service = PredictionService()
        data_fetcher = DataFetcher()
        logging.info("All services initialized successfully.")
    except Exception as e:
        logging.critical(f"FATAL: A critical service failed to initialize: {e}", exc_info=True)

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

