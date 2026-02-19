<div align="center">

<h1>📈 OptiQuant</h1>
<h3><i>Quantitative Alpha Generation · Ensemble ML · Real-Time Backtesting</i></h3>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![AWS](https://img.shields.io/badge/AWS-EC2_Deployed-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-9B59B6?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-Ensemble-yellow?style=for-the-badge)](https://catboost.ai)

<br/>

> **OptiQuant** is an end-to-end quantitative trading intelligence platform — built to do what hedge funds do: engineer signals, blend models, eliminate lookahead bias, and surface alpha.

<br/>

🔴 **[Live Demo →](http://13.61.176.157:8501)**

</div>

---

## 🧭 What Is This?

Most stock prediction tools are toy projects. OptiQuant is not.

It's a full **quantitative research pipeline** — from raw OHLCV data → feature engineering → ensemble signal generation → backtesting with institutional-grade metrics. Deployed on AWS EC2. Containerized with Docker. Built for real analysis, not vibes.

---

## 🏗️ System Architecture

```
          ┌──────────────────────────────────────────┐
          │            RAW OHLCV DATA                │
          │  (CSV Upload or Live Manual Input)       │
          └─────────────────┬────────────────────────┘
                            │
                            ▼
          ┌──────────────────────────────────────────┐
          │         DataPreprocessing.py             │
          │  ┌────────────────────────────────────┐  │
          │  │  Feature Engineering (12+ signals) │  │
          │  │  • Momentum (1d, 5d, 20d returns)  │  │
          │  │  • Volatility (rolling std)        │  │
          │  │  • RSI (14-period)                 │  │
          │  │  • Risk-Adjusted Metrics           │  │
          │  │  • Lookahead-BiasFree Construction │  │
          │  └────────────────────────────────────┘  │
          └─────────────────┬────────────────────────┘
                            │
                            ▼
          ┌──────────────────────────────────────────┐
          │          ENSEMBLE MODEL LAYER            │
          │                                          │
          │   LightGBM ──┐                           │
          │   CatBoost ──┼──► Weighted Blend ──►     │
          │   RandomForest┘     Signal Score         │
          └─────────────────┬────────────────────────┘
                            │
                            ▼
          ┌──────────────────────────────────────────┐
          │         SIGNAL REFINEMENT                │
          │  • Volatility normalization              │
          │  • Smoothing function                    │
          │  • Daily cross-sectional ranking         │
          └─────────────────┬────────────────────────┘
                            │
                            ▼
          ┌──────────────────────────────────────────┐
          │        STREAMLIT DASHBOARD               │
          │  • Backtesting engine + metrics          │
          │  • Live prediction mode                  │
          │  • Interactive performance charts        │
          └──────────────────────────────────────────┘
```

---

## ✨ Core Features

### 🤖 Ensemble ML Engine
Three powerful models blend their predictions via weighted averaging — not just picking one winner. This reduces variance and improves signal stability across different market regimes.

| Model | Role |
|---|---|
| **LightGBM** | Fast gradient boosting; excels at large-scale tabular features |
| **CatBoost** | Handles categorical splits natively; robust to outliers |
| **Random Forest** | Decorrelated trees; strong regularization via bagging |

### 📐 Feature Engineering (Lookahead-Bias-Free)
All features are constructed using **only past data** — no future values leak into predictions. This is one of the most common and fatal mistakes in quantitative research. OptiQuant handles it correctly.

Engineered signals include:
- Rolling momentum (1d, 5d, 20d returns)
- Historical volatility (rolling standard deviation)
- Relative Strength Index (14-period RSI)
- Risk-adjusted return metrics
- Volume-weighted signals

### 📊 Backtesting Engine
Upload any CSV with historical stock data and instantly evaluate the strategy's historical performance.

### ⚡ Live Prediction Mode
Input a single stock's current market data and receive a real-time **signal score** — ranked against the model's expected universe.

---

## 📈 Performance Metrics Explained

| Metric | What It Measures | Why It Matters |
|---|---|---|
| **Sharpe Ratio** | Risk-adjusted excess return | The gold standard — penalizes volatility |
| **Calmar Ratio** | Return ÷ Max Drawdown | Performance during the worst periods |
| **CAGR** | Compound Annual Growth Rate | Annualized wealth accumulation rate |
| **Max Drawdown** | Largest peak-to-trough loss | Worst-case scenario exposure |
| **Win Rate** | % of profitable days | Consistency of positive returns |

---

## 🖼️ Screenshots

**Main Dashboard**

![UI Screenshot](https://private-user-images.githubusercontent.com/218722486/468286390-25572489-43fa-4d1c-9495-877826bc63c7.png)

**Performance Metrics & Cumulative Return vs. Benchmark**

![Metrics Screenshot](https://private-user-images.githubusercontent.com/218722486/468286296-0ead78c4-1139-4ab9-9199-d20d7238089f.png)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **ML Models** | LightGBM, CatBoost, Scikit-learn (RandomForest) |
| **Data Processing** | Pandas, NumPy |
| **Frontend** | Streamlit |
| **Deployment** | Docker + AWS EC2 |
| **CI/CD** | GitHub Actions |
| **Serialization** | Joblib (`.joblib` model artifacts) |

---

## 🚀 Getting Started

### Option 1 — Run Locally

```bash
# Clone the repo
git clone https://github.com/RudraDudhat2509/OptiQuant.git
cd OptiQuant

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

### Option 2 — Docker

```bash
# Build the image
docker build -t optiquant .

# Run the container
docker run -p 8501:8501 optiquant
```

Then open `http://localhost:8501` in your browser.

---

## 📂 Project Structure

```
OptiQuant/
│
├── app.py                    # Streamlit frontend — all UI logic
├── DataPreprocessing.py      # Feature engineering pipeline
│
├── model_lgbm.joblib         # Trained LightGBM model artifact
├── model_cat.joblib          # Trained CatBoost model artifact
├── model_rf.joblib           # Trained RandomForest model artifact
│
├── notebooks/                # Research & training notebooks
├── data/                     # Sample datasets
│
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
├── .github/workflows/        # CI/CD pipeline (GitHub Actions)
└── .gitignore
```

---

## 📋 CSV Format for Backtesting

Your uploaded CSV must contain the following columns with **at least 60 trading days** per stock for reliable signal generation:

```
date, open, high, low, close, volume, Name
```

| Column | Type | Description |
|---|---|---|
| `date` | `YYYY-MM-DD` | Trading date |
| `open` | float | Opening price |
| `high` | float | Daily high |
| `low` | float | Daily low |
| `close` | float | Closing price |
| `volume` | int | Shares traded |
| `Name` | string | Ticker or stock name |

---

## ⚠️ Disclaimer

> This tool is built for **educational and research purposes only**. It does not constitute financial advice. Past backtested performance does not guarantee future results. Always do your own due diligence before making investment decisions.

---

## 🔮 Roadmap

- [ ] Add **SHAP explainability** — show which features drove each prediction
- [ ] Support **real-time data ingestion** via Yahoo Finance / Alpha Vantage API
- [ ] Portfolio-level backtesting with **position sizing and rebalancing**
- [ ] **Sector-neutral** signal construction to remove market-wide bias
- [ ] Export detailed backtest reports as PDF

---

<div align="center">

*Built with Python, gradient boosting, and a healthy obsession with avoiding lookahead bias.*

**[🔴 Try the Live Demo](http://13.61.176.157:8501)** · [Report an Issue](https://github.com/RudraDudhat2509/OptiQuant/issues)

</div>
