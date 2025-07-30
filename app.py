import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, date

# Import the feature engineering function from your separate file
from DataPreprocessing import add_features

# --- App Configuration and Title ---
st.set_page_config(
    page_title="OptiQuant Alpha Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 OptiQuant: AI-Powered Alpha")
st.markdown("""
Welcome to **OptiQuant**. This tool leverages an ensemble machine learning model to predict stock performance.
Choose your analysis mode from the sidebar.
""")

# --- About Section in a Collapsible Expander ---
with st.expander("About OptiQuant"):
    st.markdown("""
    OptiQuant is an AI-driven alpha model developed by Rudra, a DSAI student at IIT Bhilai (Class of 2028). It combines CatBoost, LightGBM, and Random Forest algorithms to generate predictive signals for stock performance. The model achieves a 63% win rate and a Sharpe ratio of 2.73, with robust backtesting capabilities. Deployed on AWS EC2, it supports both historical backtesting via CSV uploads and single-stock predictions. Explore the project on [GitHub](https://github.com/RudraDudhat2509/OptiQuant).
    """)

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Loads the pre-trained model files."""
    try:
        model1 = joblib.load('model_lgbm.joblib')
        model2 = joblib.load('model_cat.joblib')
        model3 = joblib.load('model_rf.joblib')
        return model1, model2, model3
    except FileNotFoundError:
        st.error("Model files not found! Please run your training script to generate the .joblib files and place them in the same directory as this app.")
        return None, None, None

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ User Inputs")
    input_method = st.radio(
        "Choose Analysis Mode",
        ("Upload CSV for Backtesting", "Live Prediction (Single Stock)")
    )

    uploaded_file = None
    single_stock_form = None

    if input_method == "Upload CSV for Backtesting":
        uploaded_file = st.file_uploader("Upload your stock data CSV", type=["csv"])
    else:
        st.subheader("Enter Live Stock Data")
        st.warning("This mode provides a rough estimate as it cannot calculate historical features (e.g., momentum, volatility). For full analysis, please use the CSV upload.")
        
        with st.form("single_stock_form"):
            name = st.text_input("Stock Name (e.g., AAPL)", "AAPL")
            col1, col2 = st.columns(2)
            with col1:
                open_price = st.number_input("Open Price", value=150.0, format="%.2f")
                high_price = st.number_input("High Price", value=152.0, format="%.2f")
                close_price = st.number_input("Close Price", value=151.0, format="%.2f")
            with col2:
                low_price = st.number_input("Low Price", value=149.0, format="%.2f")
                volume = st.number_input("Volume", value=1000000)
                rolling_mean_5d = st.number_input("5-Day Avg. Price", value=150.5, format="%.2f")

            submitted = st.form_submit_button("Predict Single Stock")
            if submitted:
                single_stock_form = {
                    "Name": name, "open": open_price, "high": high_price, "low": low_price,
                    "close": close_price, "volume": volume, "RollingMean_5d": rolling_mean_5d
                }

# --- Processing and Metrics Functions ---

def process_csv_upload(df_input):
    """Processes an uploaded CSV file for full backtesting analysis."""
    st.subheader("1. Data Preview")
    st.dataframe(df_input.head())

    # --- Calculate and Display Unique Stocks and Date Range ---
    unique_stocks = df_input['Name'].nunique()
    date_min = pd.to_datetime(df_input['Date']).min().strftime('%Y-%m-%d')
    date_max = pd.to_datetime(df_input['Date']).max().strftime('%Y-%m-%d')
    st.markdown(f"**Dataset Overview**")
    st.markdown(f"- Total Unique Stocks: {unique_stocks}")
    st.markdown(f"- Date Range: {date_min} to {date_max}")

    with st.spinner("Engineering features and making predictions..."):
        df_featured = add_features(df_input)

        feature_cols = ['Momentum_5d', 'Momentum_10d', 'Volatility_20d', 'Volatility_60d',
                        'AvgVol_10d', 'RSI_14', 'Momentum_5d_z', 'DownsideVol_20d',
                        'Skew_20d', 'Beta_20d', 'RollingMean_5d', 'Price_to_AvgVolume']
        
        df_predict = df_featured.dropna(subset=feature_cols).copy()
        
        if df_predict.empty:
            st.error("Error: Not enough data to make predictions.")
            st.warning("The uploaded CSV file does not contain enough historical data for each stock to calculate all necessary features (e.g., 60-day volatility). Please provide a file with at least 60 trading days of data for each stock.")
            return None

        X_predict = df_predict[feature_cols]

        model1, model2, model3 = load_models()
        if model1 is None: return None

        preds = {
            'lgbm': model1.predict(X_predict),
            'cat': model2.predict(X_predict),
            'rf': model3.predict(X_predict)
        }
        weights = {'lgbm': 0.4, 'cat': 0.4, 'rf': 0.2}

        df_predict['PredictedReturn'] = (
            weights['lgbm'] * preds['lgbm'] +
            weights['cat'] * preds['cat'] +
            weights['rf'] * preds['rf']
        )
        df_predict['PredictedReturn'] /= (df_predict['Volatility_20d'] + 1e-9)
        smoothed = df_predict.groupby('Name')['PredictedReturn'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df_predict['FinalSignal'] = smoothed
        df_predict['RankedSignal'] = df_predict.groupby('Date')['FinalSignal'].rank(pct=True)

    st.success("Analysis complete!")
    return df_predict

def predict_single_stock(data):
    """Handles prediction for a single stock with limited features."""
    volatility_20d = (data['high'] - data['low']) / data['open']
    volatility_60d = volatility_20d * 0.8
    downside_vol_20d = volatility_20d * 0.5
    price_to_avg_vol = data['close'] / (data['volume'] + 1e-9)

    features = {
        'Momentum_5d': 0.0,
        'Momentum_10d': 0.0,
        'Volatility_20d': volatility_20d,
        'Volatility_60d': volatility_60d,
        'AvgVol_10d': data['volume'],
        'RSI_14': 50,
        'Momentum_5d_z': 0.0,
        'DownsideVol_20d': downside_vol_20d,
        'Skew_20d': 0.0,
        'Beta_20d': 1.0,
        'RollingMean_5d': data['RollingMean_5d'],
        'Price_to_AvgVolume': price_to_avg_vol
    }

    X_predict = pd.DataFrame([features])

    model1, model2, model3 = load_models()
    if model1 is None:
        return

    with st.spinner("Making live prediction..."):
        preds = {
            'lgbm': model1.predict(X_predict),
            'cat': model2.predict(X_predict),
            'rf': model3.predict(X_predict)
        }

        weights = {'lgbm': 0.4, 'cat': 0.4, 'rf': 0.2}
        predicted_signal = (
            weights['lgbm'] * preds['lgbm'][0] +
            weights['cat'] * preds['cat'][0] +
            weights['rf'] * preds['rf'][0]
        )

    st.metric(
        label=f"Predicted Signal for {data['Name']}",
        value=f"{predicted_signal:.4f}",
        help="This is a raw signal score. Positive values suggest potential outperformance. This is less reliable than the full CSV analysis."
    )

def compute_metrics(df, top_k_percent=0.10):
    """Calculates key performance metrics for the backtest."""
    df = df.copy()

    daily_returns = (
        df.groupby('Date')
        .apply(lambda x: x.sort_values('RankedSignal', ascending=False)
                        .head(int(len(x) * top_k_percent))['Target'].mean())
    ).dropna()

    if daily_returns.empty:
        return {metric: 0 for metric in ['Sharpe Ratio', 'Calmar Ratio', 'Maximum Drawdown', 'CAGR', 'Win Rate']}

    cumulative_returns = (1 + daily_returns).cumprod()
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    sharpe_ratio = daily_returns.mean() / (daily_returns.std() + 1e-9) * np.sqrt(252)
    
    total_days = len(daily_returns)
    cagr = cumulative_returns.iloc[-1]**(252/total_days) - 1 if total_days > 0 else 0
    
    calmar_ratio = cagr / abs(max_drawdown + 1e-9) if max_drawdown != 0 else np.inf
    win_rate = (daily_returns > 0).mean()

    return {
        'Sharpe Ratio': sharpe_ratio,
        'Calmar Ratio': calmar_ratio,
        'Maximum Drawdown': max_drawdown,
        'CAGR': cagr,
        'Win Rate': win_rate
    }

# --- Main Display Logic ---
if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        if 'date' in df_input.columns:
            df_input['Date'] = pd.to_datetime(df_input['date'])
            df_input.drop(columns=['date'], inplace=True)
        
        df_results = process_csv_upload(df_input)

        if df_results is not None:
            st.subheader("2. Top Ranked Stocks (Latest Date)")
            latest_date = df_results['Date'].max()
            df_latest = df_results[df_results['Date'] == latest_date].sort_values('RankedSignal', ascending=False)
            
            st.write(f"Displaying top investment opportunities for **{latest_date.strftime('%Y-%m-%d')}**")
            st.dataframe(df_latest[['Name', 'close', 'FinalSignal', 'RankedSignal']].head(10).style.format({'RankedSignal': "{:.2%}"}))

            st.subheader("3. Backtest Performance")
            st.markdown("Adjust the slider to see how the strategy performs when selecting a different percentile of top-ranked stocks each day.")
            
            top_k = st.slider(
                "Select percentile of top stocks for strategy:",
                min_value=1,
                max_value=20,
                value=10,
                format="%d%%"
            )
            top_k_percent = top_k / 100.0

            st.markdown("#### Key Performance Metrics")
            metrics = compute_metrics(df_results, top_k_percent=top_k_percent)
            
            cols = st.columns(5)
            cols[0].metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
            cols[1].metric("Calmar Ratio", f"{metrics['Calmar Ratio']:.2f}")
            cols[2].metric("CAGR", f"{metrics['CAGR']:.2%}")
            cols[3].metric("Max Drawdown", f"{metrics['Maximum Drawdown']:.2%}")
            cols[4].metric("Win Rate", f"{metrics['Win Rate']:.2%}")
            
            st.markdown("#### Cumulative Strategy Return")
            
            def plot_cumulative_returns(df, top_k_percent=0.10):
                df = df.copy()
                daily_returns = (
                    df.groupby('Date')
                    .apply(lambda x: x.sort_values('RankedSignal', ascending=False)
                                    .head(int(len(x) * top_k_percent))['Target'].mean())
                ).dropna()

                if daily_returns.empty:
                    st.warning("Could not compute cumulative returns. This may be due to insufficient data to calculate future returns (Target).")
                    return None

                cum_returns = (1 + daily_returns).cumprod()
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(cum_returns.index, cum_returns.values, label=f"Top {int(top_k_percent*100)}% Strategy", color='navy')
                ax.set_xlabel("Date")
                ax.set_ylabel("Cumulative Return")
                ax.set_title("Simulated Cumulative Strategy Return")
                ax.grid(True)
                ax.legend()
                plt.tight_layout()
                return fig

            fig = plot_cumulative_returns(df_results, top_k_percent=top_k_percent)
            if fig:
                st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred processing the CSV file: {e}")

elif single_stock_form is not None:
    predict_single_stock(single_stock_form)

else:
    st.info("Please choose an analysis mode from the sidebar to get started.")
