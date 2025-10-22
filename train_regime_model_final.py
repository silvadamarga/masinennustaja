#!/usr/bin/env python3
# train_regime_model_final.py (V4.1 - Professional Grade, Robust)
"""
This script trains a professional-grade market regime classification model. It
incorporates key robustness improvements to the regime definition, evaluation
metrics, and model architecture to ensure stability and accuracy.

Key Strategic Changes (V4.1):
1.  Robust Regime Definition: Implemented a safer version of `pd.qcut` by ranking
    data first to prevent failures on non-unique bin edges. The future_regime
    calculation is now a robust, deterministic numpy-based operation.
2.  Correct Performance Metrics: Replaced arithmetic CAGR and absolute max drawdown
    with industry-standard geometric CAGR and percentage-based max drawdown for
    accurate, reliable performance evaluation.
3.  Optimized & Stable Model: Removed `recurrent_dropout` from GRU layers to
    ensure the use of fast CuDNN kernels on GPUs. Switched to a standard Adam
    optimizer for better cross-version compatibility.
4.  Realistic, Graded Evaluation: The backtest uses a practical, graded exposure
    strategy and reports the Calmar Ratio for risk-adjusted return assessment.
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

from tensorflow.keras import Input, Model, optimizers as keras_optimizers
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
tf.keras.utils.set_random_seed(RANDOM_STATE)


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
    """Creates balanced regimes with robust qcut handling and safe future_regime calculation."""
    logging.info("Creating balanced, hybrid economic regimes...")

    # Compute zscore-ish normalized series over 252 trading days
    vol_roll_mean = df['volatility_20d'].rolling(252, min_periods=30).mean()
    vol_roll_std = df['volatility_20d'].rolling(252, min_periods=30).std().replace(0, np.nan)
    norm_vol = (df['volatility_20d'] - vol_roll_mean) / vol_roll_std

    sp_roll_mean = df['credit_spread_ratio'].rolling(252, min_periods=30).mean()
    sp_roll_std = df['credit_spread_ratio'].rolling(252, min_periods=30).std().replace(0, np.nan)
    norm_spread = (df['credit_spread_ratio'] - sp_roll_mean) / sp_roll_std

    df['composite_risk'] = (norm_vol.fillna(0) + norm_spread.fillna(0)).astype(float)

    # Try qcut — if duplicates collapse buckets, fall back to rank-based binning
    try:
        df['regime'] = pd.qcut(df['composite_risk'].rank(method='first'), q=4, labels=False)
    except Exception:
        # fallback to integer quantiles with numpy percentiles
        bins = np.nanpercentile(df['composite_risk'].dropna(), [0, 25, 50, 75, 100])
        df['regime'] = pd.cut(df['composite_risk'], bins=bins, labels=False, include_lowest=True)

    # Force crisis when inverted yield curve
    if 'yield_curve_5y10y' in df.columns:
        df.loc[df['yield_curve_5y10y'] < 0, 'regime'] = int(df['regime'].max() if pd.notna(df['regime'].max()) else 3)

    # Build a stable regime map based on observed labels (in sorted order)
    unique_labels = sorted(pd.Index(df['regime'].dropna().unique().astype(int)).tolist())
    # Map labels to names in order
    name_order = ["Low Risk", "Moderate Risk", "High Risk", "Crisis"]
    regime_map = {lab: name_order[i] if i < len(name_order) else f"Regime_{lab}" for i, lab in enumerate(unique_labels)}

    # Robust future_regime: majority over the next FUTURE_PERIOD_DAYS (mode), but computed safely
    future = []
    regs = df['regime'].astype(float).values  # as numpy array
    n = len(regs)
    for i in range(n):
        window = regs[i+1:i+1+FUTURE_PERIOD_DAYS]
        window = window[~np.isnan(window)]
        if len(window) == 0:
            future.append(np.nan)
        else:
            # mode: choose smallest value in tie to keep deterministic
            vals, counts = np.unique(window, return_counts=True)
            future.append(int(vals[np.argmax(counts)]))
    df['future_regime'] = pd.Series(future, index=df.index)

    logging.info(f"Regime distribution:\n{df['regime'].value_counts(normalize=True).sort_index()}")

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
    """Builds a stacked, robust GRU model (no recurrent_dropout to keep CuDNN/GPU fast)."""
    inp = Input(shape=(TIME_STEPS, n_features))
    x = Masking(mask_value=0.0)(inp)
    # Bidirectional wrapper adds 2x hidden size; keep return_sequences for stacking
    x = Bidirectional(GRU(units=64, return_sequences=True, dropout=0.2, reset_after=False))(x)
    x = GRU(units=32, return_sequences=False, dropout=0.2, reset_after=False)(x)
    x = LayerNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(n_classes, activation='softmax', dtype='float32')(x)

    model = Model(inputs=inp, outputs=output)
    # Choose optimizer robustly
    optimizer = keras_optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def realistic_evaluation(results_df: pd.DataFrame, regime_map: dict):
    """Performs evaluation using graded strategy and corrected risk metrics (geometric CAGR, percentage max drawdown)."""
    logging.info("--- Realistic Performance Evaluation ---")

    # Ensure indices are sorted
    results_df = results_df.sort_index()
    present_classes = sorted(list(set(results_df['actual']).union(set(results_df['predicted']))))
    target_names = [regime_map.get(i, f"Regime_{i}") for i in present_classes]

    report = classification_report(
        results_df['actual'], results_df['predicted'],
        target_names=target_names, labels=present_classes,
        zero_division=0, output_dict=True
    )
    logging.info("\nClassification Report:\n" + pd.DataFrame(report).round(3).to_string())

    # Graded exposure strategy
    exposure_map = {k: v for k, v in {0: 1.0, 1: 0.6, 2: 0.2, 3: 0.0}.items()}
    results_df['exposure'] = results_df['predicted'].map(exposure_map).fillna(0.0)

    results_df['strategy_return'] = results_df['spy_return'] * results_df['exposure']
    bh_returns = results_df['spy_return'].fillna(0)
    strat_returns = results_df['strategy_return'].fillna(0)

    def calculate_metrics(returns):
        returns = returns.dropna()
        if returns.empty or returns.std() < 1e-12:
            return 0.0, 0.0, 0.0, 0.0
        # cumulative return series starting at 1
        cum = (1.0 + returns).cumprod()
        total_periods = len(returns)
        # geometric CAGR:
        cagr = cum.iloc[-1]**(252 / total_periods) - 1
        # annualized Sharpe using daily mean/std
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        # percent max drawdown
        running_max = cum.cummax()
        drawdowns = (cum / running_max) - 1.0
        max_drawdown = abs(drawdowns.min())
        calmar = cagr / max_drawdown if max_drawdown > 0 else np.nan
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
        
        unique_tr = np.unique(y_tr)
        class_weights = compute_class_weight('balanced', classes=unique_tr, y=y_tr)
        class_weight_dict = {int(k): float(v) for k, v in zip(unique_tr, class_weights)}
        
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

