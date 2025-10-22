import pandas as pd
import numpy as np
import pandas_ta as ta
import logging

def create_regime_features_for_live(market_data: dict) -> pd.DataFrame:
    """
    Prepares the feature set required for a LIVE v4 regime prediction.
    This replicates the feature engineering from the training script.
    """
    # --- Step 1: Gather all required dataframes ---
    required_tickers = ["SPY", "^VIX", "^TNX", "^FVX", "HYG", "IEF", "GLD"]
    if not all(t in market_data and market_data[t] is not None for t in required_tickers):
        missing = [t for t in required_tickers if t not in market_data or market_data[t] is None]
        logging.error(f"Missing one or more required tickers for regime prediction: {missing}")
        return None

    # --- Step 2: Combine into a single market DataFrame ---
    market_df = market_data["SPY"].copy()
    for t in required_tickers:
        clean_name = t.replace('^','') + '_Close'
        if t != "SPY":
            market_df = market_df.join(market_data[t][['Close']].rename(columns={'Close': clean_name}))

    # --- Step 3: Calculate base indicators and macro features ---
    market_df['log_return'] = np.log(market_df['Close'] / market_df['Close'].shift(1))
    market_df['volatility_20d'] = market_df['log_return'].rolling(20).std() * np.sqrt(252)
    market_df['yield_curve_5y10y'] = market_df['TNX_Close'] - market_df['FVX_Close']
    market_df['credit_spread_ratio'] = market_df['HYG_Close'] / market_df['IEF_Close']
    market_df.ffill(inplace=True); market_df.bfill(inplace=True)

    # --- Step 4: Engineer all features from the training script ---
    market_df['SMA_50_rel'] = (market_df['Close'] / market_df['Close'].rolling(50).mean()) - 1.0
    market_df['SMA_200_rel'] = (market_df['Close'] / market_df['Close'].rolling(200).mean()) - 1.0
    
    market_df['RSI_14'] = ta.rsi(close=market_df['Close'], length=14)
    
    # Momentum features
    market_df['RSI_14_mom'] = market_df['RSI_14'].diff(5)
    market_df['yield_curve_mom'] = market_df['yield_curve_5y10y'].diff(5)
    market_df['volatility_mom'] = market_df['volatility_20d'].diff(5)

    # Relative strength features
    market_df['spy_vs_bonds'] = (market_df['Close'] / market_df['IEF_Close']).pct_change(20)
    market_df['spy_vs_gold'] = (market_df['Close'] / market_df['GLD_Close']).pct_change(20)

    # --- FIX: Instead of dropping rows with any NaNs, fill them with 0 ---
    # This makes the function robust to shorter histories of some indicators.
    # The model's Masking layer is designed to handle these zeros.
    market_df.fillna(0, inplace=True)
    
    return market_df


def process_data_for_prediction(df, spy_df, predicted_regime: int, features_list: list) -> pd.DataFrame:
    """
    Applies feature engineering for the breakout model, now including the predicted regime.
    """
    if df is None: return None
    
    df = df.copy().sort_index()
    spy_df_renamed = spy_df.copy()[['Close']].rename(columns={'Close': 'SPY_Close'})
    df = df.merge(spy_df_renamed, left_index=True, right_index=True, how='left')
    
    # --- Add the predicted regime as a feature ---
    if predicted_regime is not None:
        df['predicted_regime'] = predicted_regime
    else: # Handle case where regime prediction might fail
        df['predicted_regime'] = -1 # Use a neutral/error value
        
    df.ffill(inplace=True)
    close = df['Close']
    
    df['SMA_50_rel'] = (close / close.rolling(50).mean()) - 1.0
    df['ROC_20'] = close.pct_change(20)
    df['Volume_rel'] = (df['Volume'] / df['Volume'].rolling(20).mean()) - 1.0
    
    raw_atr = ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=14)

    if raw_atr is not None and not raw_atr.empty:
        if isinstance(raw_atr, pd.DataFrame):
            raw_atr = raw_atr.iloc[:, 0]
        df['ATR_14_rel'] = raw_atr / close
    else:
        df['ATR_14_rel'] = np.nan
        logging.warning(f"Could not calculate ATR for a stock (likely insufficient data). Filling with NaN.")

    # Ensure all required columns exist
    for feature in features_list:
        if feature not in df.columns:
            df[feature] = 0
            
    df.dropna(inplace=True)
    return df[features_list]

