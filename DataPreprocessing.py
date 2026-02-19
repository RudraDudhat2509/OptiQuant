import pandas as pd
import numpy as np

def compute_rsi(series: pd.Series, window: int = 14):
    """Computes the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def downside_vol(x):
    """Computes downside volatility."""
    x = x.dropna()
    return np.sqrt(np.mean(np.square(np.minimum(x, 0)))) if len(x) > 0 else np.nan

def add_features(df):
    """
    Engineers all necessary predictive features without data leakage.
    """
    df = df.copy()
    df = df.sort_values(['Name', 'Date'])

    def build_features(group):
        # --- Momentum Features ---
        group['Ret_1d'] = group['close'].pct_change()
        group['Momentum_5d'] = group['close'].pct_change(periods=5)
        group['Momentum_10d'] = group['close'].pct_change(periods=10)

        # --- Volatility Features ---
        group['Volatility_20d'] = group['Ret_1d'].rolling(20).std()
        group['Volatility_60d'] = group['Ret_1d'].rolling(60).std()
        group['DownsideVol_20d'] = group['Ret_1d'].rolling(20).apply(downside_vol, raw=False)

        # --- Volume and Liquidity Features ---
        group['AvgVol_10d'] = group['volume'].rolling(10).mean()
        group['Price_to_AvgVolume'] = group['close'] / (group['volume'].rolling(10).mean() + 1e-9)

        # --- Technical Indicators ---
        group['RSI_14'] = compute_rsi(group['close'], window=14)
        group['Skew_20d'] = group['Ret_1d'].rolling(20).skew()
        group['RollingMean_5d'] = group['close'].rolling(5).mean()
        
        return group

    df = df.groupby('Name', group_keys=False).apply(build_features)
    
    # --- Cross-sectional Features ---
    if not df.empty:
        df['Momentum_5d_z'] = df.groupby('Date')['Momentum_5d'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

    # --- HISTORICAL Market Calculation (NO LEAKAGE) ---
    df['HistoricalMarketReturn'] = df.groupby('Date')['Ret_1d'].transform('mean')

    df['Beta_20d'] = df.groupby('Name', group_keys=False).apply(
        lambda group: group['Ret_1d'].rolling(20).corr(group['HistoricalMarketReturn'])
    ).reset_index(drop=True)

    # --- Target Calculation ---
    # Target is strictly separated from feature calculation
    df['Target'] = df.groupby('Name')['close'].shift(-5) / df['close'] - 1
    df['Target'] = df['Target'].clip(-0.15, 0.15) 
    df['Target_demeaned'] = df.groupby('Date')['Target'].transform(lambda x: x - x.mean())

    return df

