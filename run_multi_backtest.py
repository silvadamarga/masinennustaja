#!/usr/bin/env python3
# train_regime_model_final.py (V4 - Professional Grade, Balanced)
"""
This script trains a professional-grade market regime classification model. It
has been completely redesigned to produce a balanced and economically logical
regime definition, a robust model architecture, and a realistic evaluation.

Key Strategic Changes:
1.  Balanced, Hybrid Regimes: Abandons unstable quantile logic and instead uses
    a composite risk score (volatility + credit spreads) to create four
    explicitly balanced buckets (quartiles) using `pd.qcut`. A sharp,
    rule-based override for an inverted yield curve ensures "Crisis" events
    are always captured. This guarantees a balanced and logical regime set.
2.  Stacked GRU Architecture: Replaces the previous complex architecture with a
    simpler, more robust stacked GRU model. This provides depth and improves
    training stability.
3.  Relative Strength Features: Adds new features comparing the performance of
    stocks to bonds (SPY/IEF) and gold (SPY/GLD), which are powerful indicators
    of risk-on/risk-off sentiment.
4.  Realistic, Graded Evaluation: The backtest now uses a more practical,
    graded exposure strategy (e.g., 100% equity in Bull, 50% in Growth, etc.)
    and reports the Calmar Ratio (CAGR / Max Drawdown) for a better assessment
    of risk-adjusted returns.
"""

import os
import logging
import pickle
import random
import json
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

# Suppress excessive TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import Input, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, LayerNormalization, Masking
from tensorflow.keras.utils import to_categorical

# ---------------- Config ----------------
HISTORY_PERIOD_YEARS = 15
TIME_STEPS = 60
FUTURE_PERIOD_DAYS = 5
RANDOM_STATE = 42
BATCH_SIZE = 64
CACHE_DIR = "data_cache_prod_v4"

# --- Output Filenames ---
MODEL_FILENAME = "market_regime_model_v4.keras"
SCALER_FILENAME = "market_regime_scaler_v4.pkl"
METADATA_FILENAME = "market_regime_metadata_v4.pkl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
tf.get_logger().setLevel('ERROR')

# ---------------- Data & Feature Helpers ----------------
def get_data(ticker: str, period_years: int) -> pd.DataFrame:
    """Fetches and caches historical data from yfinance."""
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_{period_years}y.pkl")
    if os.path.exists(cache_path):
        try: return pd.read_pickle(cache_path)
        except Exception: pass
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=f"{period_years}y", auto_adjust=True)
        if df.empty: return None
        df.index = df.index.tz_localize(None)
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_pickle(cache_path)
        return df.rename_axis('Date')
    except Exception as e:
        logging.warning(f"Could not fetch {ticker}: {e}")
        return None

def create_balanced_hybrid_regimes(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Creates balanced regimes using a composite risk score and rule-based overrides."""
    logging.info("Creating balanced, hybrid economic regimes...")
    
    # Normalize volatility and credit spreads to create a composite risk score
    norm_vol = (df['volatility_20d'] - df['volatility_20d'].rolling(252).mean()) / df['volatility_20d'].rolling(252).std()
    norm_spread = (df['credit_spread_ratio'] - df['credit_spread_ratio'].rolling(252).mean()) / df['credit_spread_ratio'].rolling(252).std()
    df['composite_risk'] = norm_vol + norm_spread
    
    # Use qcut to create 4 balanced buckets based on risk score
    df['regime'] = pd.qcut(df['composite_risk'], 4, labels=False, duplicates='drop')
    
    # Rule-based override: an inverted yield curve is ALWAYS a 'Crisis' regime
    df.loc[df['yield_curve_5y10y'] < 0, 'regime'] = 3
    
    regime_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk", 3: "Crisis"}
    logging.info(f"Regime distribution:\n{df['regime'].value_counts(normalize=True).sort_index()}")

    df['future_regime'] = df['regime'].shift(-FUTURE_PERIOD_DAYS).rolling(
        window=FUTURE_PERIOD_DAYS, min_periods=1
    ).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan, raw=False)
    
    return df, regime_map

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers base, momentum, and relative strength features."""
    df['SMA_50_rel'] = (df['Close'] / df['Close'].rolling(50).mean()) - 1.0
    df['SMA_200_rel'] = (df['Close'] / df['Close'].rolling(200).mean()) - 1.0
    df.ta.rsi(length=14, append=True)
    
    # Momentum features
    df['RSI_14_mom'] = df['RSI_14'].diff(5)
    df['yield_curve_mom'] = df['yield_curve_5y10y'].diff(5)
    
    # Relative strength features
    df['spy_vs_bonds'] = (df['Close'] / df['IEF_Close']).pct_change(20)
    df['spy_vs_gold'] = (df['Close'] / df['GLD_Close']).pct_change(20)
    
    return df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], errors='ignore')

# ---------------- Model & Sequence Helpers ----------------
def create_sequences(df: pd.DataFrame, features: List[str]):
    """Creates sequences for the RNN model."""
    data = df[features].ffill().fillna(0).values
    labels = df['future_regime'].values
    X, y, dates = [], [], []
    for i in range(TIME_STEPS, len(df)):
        if pd.notna(labels[i]):
            X.append(data[i-TIME_STEPS:i])
            y.append(int(labels[i]))
            dates.append(df.index[i])
    if not X: raise ValueError("No valid sequences created")
    return np.array(X), np.array(y), np.array(dates)

def build_stacked_gru_model(n_features: int, n_classes: int):
    """Builds a stacked, robust GRU model."""
    inp = Input(shape=(TIME_STEPS, n_features))
    x = Masking(mask_value=0.0)(inp)
    x = Bidirectional(GRU(units=64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, reset_after=False))(x)
    x = GRU(units=32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, reset_after=False)(x)
    x = LayerNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(n_classes, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs=inp, outputs=output)
    optimizer = optimizers.AdamW(learning_rate=0.001, weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def realistic_evaluation(results_df: pd.DataFrame, regime_map: dict):
    """Performs evaluation using a graded strategy and advanced risk metrics."""
    logging.info("--- Realistic Performance Evaluation ---")
    all_classes = np.arange(len(regime_map))
    
    report = classification_report(
        results_df['actual'], results_df['predicted'],
        target_names=list(regime_map.values()), labels=all_classes,
        zero_division=0, output_dict=True
    )
    logging.info("\nClassification Report:\n" + pd.DataFrame(report).round(3).to_string())

    # Graded exposure strategy
    exposure_map = {0: 1.0, 1: 0.6, 2: 0.2, 3: 0.0} # 100% in Bull, 60% in Growth, etc.
    results_df['exposure'] = results_df['predicted'].map(exposure_map)
    results_df['strategy_return'] = results_df['spy_return'] * results_df['exposure']
    
    bh_returns = results_df['spy_return'].fillna(0)
    strat_returns = results_df['strategy_return'].fillna(0)
    
    def calculate_metrics(returns):
        if returns.std() < 1e-8: return 0, 0, 0, 0
        cagr = (1 + returns.mean())**252 - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        cum_returns = (1 + returns).cumprod()
        max_drawdown = (cum_returns.cummax() - cum_returns).max()
        calmar = cagr / max_drawdown if max_drawdown > 0 else 0
        return cagr, sharpe, max_drawdown, calmar

    bh_cagr, bh_sharpe, bh_dd, bh_calmar = calculate_metrics(bh_returns)
    st_cagr, st_sharpe, st_dd, st_calmar = calculate_metrics(strat_returns)

    logging.info(f"\n--- Strategy Performance ---")
    logging.info(f"{'Metric':<15} | {'Buy & Hold':<15} | {'Regime Strategy':<15}")
    logging.info(f"{'-'*50}")
    logging.info(f"{'CAGR':<15} | {bh_cagr:<15.2%} | {st_cagr:<15.2%}")
    logging.info(f"{'Sharpe Ratio':<15} | {bh_sharpe:<15.2f} | {st_sharpe:<15.2f}")
    logging.info(f"{'Max Drawdown':<15} | {bh_dd:<15.2%} | {st_dd:<15.2%}")
    logging.info(f"{'Calmar Ratio':<15} | {bh_calmar:<15.2f} | {st_calmar:<15.2f}")

    plt.figure(figsize=(15, 8))
    (1 + bh_returns).cumprod().plot(label='Buy & Hold', linewidth=2)
    (1 + strat_returns).cumprod().plot(label='Regime Strategy', linewidth=2, grid=True)
    plt.title('Strategy Performance'); plt.ylabel('Cumulative Return'); plt.legend()
    plt.savefig('equity_curve_v4.png'); plt.close()

# ---------------- Main Execution Logic ----------------
def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e: logging.error(f"Could not configure GPU: {e}")

    tf.random.set_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE)
    
    logging.info("Fetching all required market data...")
    tickers = ["SPY", "^VIX", "^TNX", "^FVX", "HYG", "IEF", "GLD"]
    all_data = {t: get_data(t, HISTORY_PERIOD_YEARS) for t in tickers}
    if any(df is None for df in all_data.values()):
        logging.error("Failed to fetch critical market data. Halting."); return
        
    market_df = all_data["SPY"].copy()
    for t in tickers:
        clean_name = t.replace('^','') + '_Close'
        if t != "SPY": market_df = market_df.join(all_data[t][['Close']].rename(columns={'Close': clean_name}))
    
    market_df['log_return'] = np.log(market_df['Close'] / market_df['Close'].shift(1))
    market_df['volatility_20d'] = market_df['log_return'].rolling(20).std() * np.sqrt(252)
    market_df['yield_curve_5y10y'] = market_df['TNX_Close'] - market_df['FVX_Close']
    market_df['credit_spread_ratio'] = market_df['HYG_Close'] / market_df['IEF_Close']
    market_df.ffill(inplace=True); market_df.bfill(inplace=True)

    market_df, regime_map = create_balanced_hybrid_regimes(market_df)
    spy_close_for_backtest = market_df['Close'].copy()
    market_df = create_features(market_df)
    market_df.dropna(subset=['future_regime'], inplace=True)
    
    all_features = list(market_df.select_dtypes(include=np.number).columns)
    exclude = ['regime', 'future_regime', 'log_return', 'composite_risk']
    FEATURES = [c for c in all_features if c not in exclude]
    logging.info(f"Using {len(FEATURES)} features for training.")
    
    logging.info("Starting Walk-Forward Cross-Validation...")
    all_dates = sorted(market_df.index.unique())
    start_date, end_date_limit = all_dates[0], all_dates[-1] - pd.DateOffset(months=6)
    current_date, fold, all_preds = start_date, 1, []
    
    while current_date + pd.DateOffset(years=7) < end_date_limit:
        train_end = current_date + pd.DateOffset(years=7)
        test_end = train_end + pd.DateOffset(months=6)
        logging.info(f"--- FOLD {fold}: Training -> {train_end.date()}, Testing -> {test_end.date()} ---")
        
        train_df = market_df.loc[current_date:train_end]
        test_df = market_df.loc[train_end + pd.DateOffset(days=1):test_end]

        if len(train_df) < 252 or len(test_df) < 60:
            current_date += pd.DateOffset(months=6); continue
        
        X_tr, y_tr, _ = create_sequences(train_df, FEATURES)
        X_ts, y_ts, dates_ts = create_sequences(test_df, FEATURES)
        
        scaler = MinMaxScaler().fit(X_tr.reshape(-1, X_tr.shape[-1]))
        X_tr_s = scaler.transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
        X_ts_s = scaler.transform(X_ts.reshape(-1, X_ts.shape[-1])).reshape(X_ts.shape)
        
        model = build_stacked_gru_model(len(FEATURES), len(regime_map))
        y_tr_cat = to_categorical(y_tr, num_classes=len(regime_map))
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
        class_weight_dict = dict(zip(np.unique(y_tr), class_weights))
        
        model.fit(X_tr_s, y_tr_cat, epochs=60, batch_size=BATCH_SIZE, validation_split=0.15,
                  class_weight=class_weight_dict,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=0)
        
        probs = model.predict(X_ts_s)
        fold_results = pd.DataFrame({
            'date': dates_ts, 'actual': y_ts, 'predicted': np.argmax(probs, axis=1),
            **{f'prob_{i}': probs[:, i] for i in range(len(regime_map))}
        })
        all_preds.append(fold_results)
        current_date += pd.DateOffset(months=6); fold += 1
        
    if not all_preds:
        logging.error("Walk-forward validation produced no results. Halting."); return

    results_df = pd.concat(all_preds).drop_duplicates('date').set_index('date')
    results_df['spy_return'] = spy_close_for_backtest.pct_change()
    realistic_evaluation(results_df, regime_map)
    
    logging.info("Retraining final model on all data for deployment...")
    X_all, y_all, _ = create_sequences(market_df, FEATURES)
    scaler_final = MinMaxScaler().fit(X_all.reshape(-1, X_all.shape[-1]))
    X_all_s = scaler_final.transform(X_all.reshape(-1, X_all.shape[-1])).reshape(X_all.shape)
    y_all_cat = to_categorical(y_all, num_classes=len(regime_map))
    
    final_model = build_stacked_gru_model(len(FEATURES), len(regime_map))
    final_model.fit(X_all_s, y_all_cat, epochs=30, batch_size=BATCH_SIZE, verbose=2)
    
    final_model.save(MODEL_FILENAME)
    with open(SCALER_FILENAME, 'wb') as f: pickle.dump(scaler_final, f)
    metadata = {'features': FEATURES, 'regime_map': regime_map, 'time_steps': TIME_STEPS}
    with open(METADATA_FILENAME, 'wb') as f: pickle.dump(metadata, f)

    logging.info("Process complete. Production model v4 and artifacts saved.")

if __name__ == "__main__":
    main()

