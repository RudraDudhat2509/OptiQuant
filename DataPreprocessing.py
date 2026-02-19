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
    Refactored to use high-performance vectorized transforms, silencing Pandas FutureWarnings.
    """
    df = df.copy()
    df = df.sort_values(['Name', 'Date'])

    # Create a groupby object once to reuse across vectorized transformations
    grouped = df.groupby('Name')

    # --- Momentum Features ---
    df['Ret_1d'] = grouped['close'].pct_change()
    df['Momentum_5d'] = grouped['close'].pct_change(periods=5)
    df['Momentum_10d'] = grouped['close'].pct_change(periods=10)

    # --- Volatility Features ---
    df['Volatility_20d'] = grouped['Ret_1d'].transform(lambda x: x.rolling(20).std())
    df['Volatility_60d'] = grouped['Ret_1d'].transform(lambda x: x.rolling(60).std())
    df['DownsideVol_20d'] = grouped['Ret_1d'].transform(lambda x: x.rolling(20).apply(downside_vol, raw=False))

    # --- Volume and Liquidity Features ---
    df['AvgVol_10d'] = grouped['volume'].transform(lambda x: x.rolling(10).mean())
    df['Price_to_AvgVolume'] = df['close'] / (df['AvgVol_10d'] + 1e-9)

    # --- Technical Indicators ---
    df['RSI_14'] = grouped['close'].transform(lambda x: compute_rsi(x, window=14))
    df['Skew_20d'] = grouped['Ret_1d'].transform(lambda x: x.rolling(20).skew())
    df['RollingMean_5d'] = grouped['close'].transform(lambda x: x.rolling(5).mean())
    
    # --- Cross-sectional Features ---
    if not df.empty:
        df['Momentum_5d_z'] = df.groupby('Date')['Momentum_5d'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )

    # --- HISTORICAL Market Calculation (NO LEAKAGE) ---
    df['HistoricalMarketReturn'] = df.groupby('Date')['Ret_1d'].transform('mean')

    # --- Beta Calculation (Warning Fixed) ---
    # We explicitly pass include_groups=False to silence the Pandas deprecation warning
    # while calculating the rolling correlation across two distinct columns.
    df['Beta_20d'] = df.groupby('Name', group_keys=False).apply(
        lambda g: g['Ret_1d'].rolling(20).corr(g['HistoricalMarketReturn']),
        include_groups=False
    )

    # --- Target Calculation ---
    df['Target'] = grouped['close'].shift(-5) / df['close'] - 1
    df['Target'] = df['Target'].clip(-0.15, 0.15) 
    df['Target_demeaned'] = df.groupby('Date')['Target'].transform(lambda x: x - x.mean())

    return df
