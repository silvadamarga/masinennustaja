import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import logging
import os

# --- Local Imports ---
from feature_engineering import process_data_for_prediction, create_regime_features_for_live

class PredictionService:
    """
    A production-ready service to encapsulate BOTH the breakout and the V4 regime ML models.
    This version includes robust error handling for graceful startup and configurable model paths.
    """
    def __init__(self,
                 breakout_model_path=None,
                 breakout_scaler_path=None,
                 breakout_metadata_path=None,
                 regime_model_path=None,
                 regime_scaler_path=None,
                 regime_metadata_path=None):
        """
        Loads all necessary artifacts for both models. If any artifact fails to load,
        the service will enter a 'not ready' state but will not crash the application.
        """
        logging.info("Initializing PredictionService for production...")

        self.is_ready = False # Service is not ready until all models are loaded
        
        # --- Model Paths (configurable via environment variables or defaults) ---
        self.breakout_model_path = breakout_model_path or os.getenv('BREAKOUT_MODEL_PATH', 'final_model_walkforward.keras')
        self.breakout_scaler_path = breakout_scaler_path or os.getenv('BREAKOUT_SCALER_PATH', 'final_scaler_walkforward.pkl')
        self.breakout_metadata_path = breakout_metadata_path or os.getenv('BREAKOUT_METADATA_PATH', 'final_metadata_walkforward.pkl')
        self.regime_model_path = regime_model_path or os.getenv('REGIME_MODEL_PATH', 'market_regime_model_v4.keras')
        self.regime_scaler_path = regime_scaler_path or os.getenv('REGIME_SCALER_PATH', 'market_regime_scaler_v4.pkl')
        self.regime_metadata_path = regime_metadata_path or os.getenv('REGIME_METADATA_PATH', 'market_regime_metadata_v4.pkl')

        try:
            # --- Load Breakout Model Artifacts ---
            logging.info(f"Loading breakout model from: {self.breakout_model_path}")
            self.breakout_model = tf.keras.models.load_model(self.breakout_model_path, safe_mode=False)
            with open(self.breakout_scaler_path, 'rb') as f: self.breakout_scaler = pickle.load(f)
            with open(self.breakout_metadata_path, 'rb') as f: self.breakout_features = pickle.load(f)['features']
            
            # --- Load V4 Regime Model Artifacts ---
            logging.info(f"Loading regime model from: {self.regime_model_path}")
            self.regime_model = tf.keras.models.load_model(self.regime_model_path, safe_mode=False)
            with open(self.regime_scaler_path, 'rb') as f: self.regime_scaler = pickle.load(f)
            with open(self.regime_metadata_path, 'rb') as f:
                regime_meta = pickle.load(f)
                self.regime_features = regime_meta['features']
                self.regime_map = regime_meta['regime_map']
                # Load time_steps from metadata to ensure consistency
                self.time_steps = regime_meta.get('time_steps', 60)

            # --- Configuration ---
            self.confidence_threshold = 0.65
            self.bullish_regimes = [0] # V4 model defines regime 0 as "Low Risk"
            
            self.is_ready = True
            logging.info("All model artifacts loaded successfully. Service is ready.")

        except FileNotFoundError as e:
            logging.error(f"FATAL: A required model artifact was not found: {e}. The service will not be available.")
        except Exception as e:
            logging.error(f"FATAL: An unexpected error occurred during model loading: {e}", exc_info=True)
            # The service remains in a 'not ready' state

    def predict_current_regime(self, market_data: dict) -> dict:
        """Predicts the current market regime and returns probabilities."""
        if not self.is_ready: return {"error": "Regime model is not available."}
        try:
            market_df = create_regime_features_for_live(market_data)
            if market_df is None or len(market_df) < self.time_steps:
                return {"error": "Not enough data for regime prediction."}

            sequence = market_df[self.regime_features].tail(self.time_steps).values
            sequence_scaled = self.regime_scaler.transform(sequence.reshape(-1, len(self.regime_features))).reshape(1, self.time_steps, -1)
            
            prediction = self.regime_model.predict(sequence_scaled, verbose=0)
            regime_id = np.argmax(prediction)
            regime_name = self.regime_map.get(str(regime_id), self.regime_map.get(regime_id, "Unknown"))
            
            return {
                "regime_id": int(regime_id),
                "regime_name": regime_name,
                "is_bullish": int(regime_id) in self.bullish_regimes,
                "probabilities": prediction[0].tolist()
            }
        except Exception as e:
            logging.error(f"Regime prediction failed: {e}", exc_info=True)
            return {"error": "Regime prediction failed."}

    def prepare_data_for_prediction(self, ticker, price_df, spy_df, predicted_regime: int):
        """Prepares a single stock's data, including the predicted regime as a feature."""
        if not self.is_ready:
             return {"Ticker": ticker, "Price": "N/A", "Signal_Raw": "Service Unavailable", "Signal_Filtered": "Service Unavailable"}

        if price_df is None or price_df.empty:
            return {"Ticker": ticker, "Price": "N/A", "Signal_Raw": "Data Error", "Signal_Filtered": "Data Error"}

        last_price = f"${price_df['Close'].iloc[-1]:.2f}"
        try:
            processed_df = process_data_for_prediction(price_df, spy_df, predicted_regime, self.breakout_features)
            
            if processed_df is None or len(processed_df) < self.time_steps:
                return {"Ticker": ticker, "Price": last_price, "Signal_Raw": "Not Enough Data", "Signal_Filtered": "Not Enough Data"}
            
            sequence = processed_df.tail(self.time_steps).values
            sequence_scaled = self.breakout_scaler.transform(sequence)
            
            return {"type": "data", "Ticker": ticker, "Price": last_price, "sequence": sequence_scaled}
        except Exception as e:
            logging.error(f"Data preparation failed for {ticker}: {e}", exc_info=True)
            return {"Ticker": ticker, "Price": last_price, "Signal_Raw": "Prep Error", "Signal_Filtered": "Prep Error"}

    def run_batch_prediction(self, prepared_data_list, current_regime: int):
        """Runs batch prediction and returns BOTH a raw and a filtered signal."""
        if not self.is_ready:
             # Return error state for all valid data entries
             return [d for d in prepared_data_list if d.get("type") != "data"] + \
                    [{"Ticker": d["Ticker"], "Price": d["Price"], "Signal_Raw": "Service Unavailable", "Signal_Filtered": "Service Unavailable"} 
                     for d in prepared_data_list if d.get("type") == "data"]
        
        valid_data = [d for d in prepared_data_list if d.get("type") == "data"]
        error_results = [d for d in prepared_data_list if d.get("type") != "data"]

        if not valid_data: return error_results
            
        sequences_batch = np.array([d['sequence'] for d in valid_data])
        pred_bk_batch, pred_ret_batch = self.breakout_model.predict(sequences_batch, verbose=0)

        prediction_results = []
        for i, data in enumerate(valid_data):
            confidence = pred_bk_batch[i][0]
            is_breakout_signal = confidence > self.confidence_threshold
            
            signal_raw = "Potential Breakout" if is_breakout_signal else "Hold"
            signal_filtered = signal_raw
            
            # Only apply filter if the regime was successfully predicted
            if current_regime is not None and is_breakout_signal and (current_regime not in self.bullish_regimes):
                signal_filtered = "Hold (Regime Filter)"

            prediction_results.append({
                "Ticker": data['Ticker'],
                "Price": data['Price'],
                "Confidence": f"{confidence:.2%}",
                "Pred_Return": f"{pred_ret_batch[i][0]:.2f}%",
                "Signal_Raw": signal_raw,
                "Signal_Filtered": signal_filtered
            })
        return prediction_results + error_results


