# 📈 OptiQuant: AI-Powered Alpha (v2.0 Architecture)

**OptiQuant** is an end-to-end quantitative analysis tool that leverages a sophisticated ensemble machine learning model (LightGBM, CatBoost, RandomForest) to predict stock performance. 

*Note: Version 2.0 represents a complete architectural rewrite focusing on mathematical rigor, vectorized performance, and the strict prevention of future data leakage.*

---

### 🚀 Live Demo

[Access the Live AWS Deployment Here](http://13.61.176.157:8501)

---

### ✨ v2.0 Production Features

-   **Strict Lookahead Bias Prevention:** Target variables and predictive features are strictly decoupled. Cross-sectional metrics like Beta and momentum are calculated purely on historical $T-0$ data to prevent future data leakage.
-   **Vectorized Feature Engineering:** Ripped out slow Pandas `.apply()` loops in favor of C-level `.transform()` operations, massively speeding up the inference and feature generation pipeline.
-   **Trading Frictions Engine:** Added dynamic UI controls to penalize the model with real-world transaction costs and slippage (in basis points) dynamically calculated against daily portfolio turnover.
-   **Deterministic CI/CD Testing:** Integrated a `pytest` suite to mathematically prove the absence of data leakage before any model retraining.
-   **Ensemble ML Model:** Utilizes a weighted blend of LightGBM, CatBoost, and RandomForest for robust and accurate predictions.
-   **Interactive Performance Analysis:** Adjust strategy deciles and analyze daily-rebalanced metrics instantly.

---

### 📊 Realistic Performance Metrics (Net of Costs)

Unlike academic models that ignore market realities, OptiQuant v2.0 evaluates performance **net of trading frictions**. 

*Metrics below represent a daily-rebalanced, top-decile portfolio net of 10 bps transaction costs and 5 bps slippage:*
-   **Realistic CAGR:** ~25%
-   **Sharpe Ratio:** ~1.05
-   **Win Rate:** ~53%
-   **Maximum Drawdown:** Mathematically calculated via continuous peak-to-trough analysis.

---

### 🛠️ Tech Stack

-   **Backend & Modeling:** Python, Pandas, NumPy, Scikit-learn, LightGBM, CatBoost
-   **Testing & Safety:** Pytest, GitHub Actions CI/CD
-   **Frontend:** Streamlit
-   **Deployment:** Docker, AWS EC2

---

### 🧠 Methodology

The core of OptiQuant is a predictive model trained to identify stocks that are likely to outperform the market average over the next 5 trading days.

1.  **Feature Engineering:** Raw OHLCV data is transformed into a rich feature set using highly optimized, vectorized transformations. 
2.  **Ensemble Modeling:** Three powerful gradient-boosting and tree-based models are trained on the engineered features. Their predictions are combined using a weighted average to produce a stable forecast.
3.  **Cost-Adjusted Signal Generation:** The raw model output is refined into a final signal. During backtesting, the strategy calculates daily turnover and mathematically deducts basis points for slippage and trading costs to reflect true market execution.

---

### 🚀 How to Use the App

1.  Navigate to the live demo link.
2.  From the sidebar, choose your analysis mode:
    -   **For Backtesting:** Select "Upload CSV for Backtesting" and upload a CSV file containing historical stock data. The file must include `date`, `open`, `high`, `low`, `close`, `volume`, and `Name` columns.
    -   **For a Quick Prediction:** Select "Live Prediction (Single Stock)" and fill in the form with the stock's current data.
3.  Adjust the **Trading Frictions** (bps) in the sidebar to stress-test the model's profitability under different broker conditions.

---

### 🔧 Local Setup & Deployment

To run this project on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/RudraDudhat2509/OptiQuant.git](https://github.com/RudraDudhat2509/OptiQuant.git)
    cd OptiQuant
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Test Suite (Ensure Architecture Integrity):**
    ```bash
    pytest pytester.py -v
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

5.  **Build with Docker (Optional):**
    ```bash
    docker build -t optiquant-app .
    docker run -p 8501:8501 optiquant-app
    ```

---

### 📂 Project Structure


.
├── 📄 app.py                  # Main Streamlit application file
├── 📄 DataPreprocessing.py      # Vectorized feature engineering module
├── 📄 pytester.py               # Deterministic unit tests for data leakage
├── 📄 Dockerfile               # Instructions for building the Docker container
├── 📄 requirements.txt          # Python dependencies
├── 📦 model_lgbm.joblib        # Trained LightGBM model
├── 📦 model_cat.joblib         # Trained CatBoost model
└── 📦 model_rf.joblib           # Trained RandomForest model


---

### 🖼️ Screenshots
**UI** <img width="2239" height="1248" alt="image" src="https://github.com/user-attachments/assets/25572489-43fa-4d1c-9495-877826bc63c7" />

**Performance Metrics & Graph (Net of Frictions):**
<img width="1174" height="550" alt="image" src="https://github.com/user-attachments/assets/0ead78c4-1139-4ab9-9199-d20d7238089f" />
