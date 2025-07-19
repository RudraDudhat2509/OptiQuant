# OptiQuant
# 📈 OptiQuant: AI-Powered Alpha

**OptiQuant** is an end-to-end quantitative analysis tool that leverages a sophisticated ensemble machine learning model to predict stock performance and identify potential investment opportunities. This application provides a user-friendly Streamlit interface for backtesting trading strategies and getting live predictions.

---

### 🚀 Live Demo

**[Insert Your Streamlit Community Cloud Link Here]**

---

### ✨ Features

-   **Ensemble ML Model:** Utilizes a weighted blend of LightGBM, CatBoost, and RandomForest for robust and accurate predictions.
-   **Advanced Feature Engineering:** Generates over a dozen predictive features, including momentum, volatility, RSI, and risk-adjusted metrics.
-   **Backtesting Engine:** Upload historical stock data (CSV) to simulate and evaluate the model's performance over time.
-   **Live Prediction Mode:** Get a real-time signal score for a single stock by inputting its current data.
-   **Interactive Performance Analysis:**
    -   View a ranked list of top-performing stocks for any given day in your dataset.
    -   Analyze key performance metrics like Sharpe Ratio, Calmar Ratio, CAGR, and Max Drawdown.
    -   Visualize the strategy's cumulative return against a market benchmark.
-   **Dynamic Controls:** Use an interactive slider to adjust the strategy (e.g., top 1%, 5%, 10%) and see how it impacts performance metrics and graphs instantly.

---

### 📊 Key Performance Metrics

The application calculates several industry-standard metrics to evaluate the performance of the backtested strategy:

-   **Sharpe Ratio:** Measures the risk-adjusted return, indicating how much excess return you receive for the extra volatility you endure.
-   **Calmar Ratio:** Measures return relative to the maximum drawdown, highlighting performance during the worst periods.
-   **CAGR (Compound Annual Growth Rate):** The annualized rate of return that an investment provides over a period.
-   **Maximum Drawdown:** The largest peak-to-trough decline in the value of the portfolio, representing the worst-case loss.
-   **Win Rate:** The percentage of days where the strategy yielded a positive return.

---

### 🛠️ Tech Stack

-   **Backend & Modeling:** Python, Pandas, NumPy, Scikit-learn, LightGBM, CatBoost
-   **Frontend:** Streamlit
-   **Deployment:** Docker, Streamlit Community Cloud / AWS EC2

---

### 🧠 Methodology

The core of OptiQuant is a predictive model trained to identify stocks that are likely to outperform the market average over the next 5 trading days (generating "alpha").

1.  **Feature Engineering:** Raw OHLCV data is transformed into a rich feature set designed to capture market dynamics. A crucial part of this process is the careful avoidance of **lookahead bias**, ensuring that all predictive features are calculated using only information that would have been available at the time of prediction.
2.  **Ensemble Modeling:** Three powerful gradient-boosting and tree-based models are trained on the engineered features. Their predictions are then combined using a weighted average to produce a more stable and reliable forecast.
3.  **Signal Generation:** The raw model output is refined into a final signal by normalizing for volatility and applying a smoothing function. Stocks are then ranked daily based on this final signal to identify the top investment opportunities.

---

### 🚀 How to Use the App

1.  Navigate to the live demo link.
2.  From the sidebar, choose your analysis mode:
    -   **For Backtesting:** Select "Upload CSV for Backtesting" and upload a CSV file containing historical stock data. The file must include `date`, `open`, `high`, `low`, `close`, `volume`, and `Name` columns, with at least 60 days of data per stock.
    -   **For a Quick Prediction:** Select "Live Prediction (Single Stock)" and fill in the form with the stock's current data.
3.  Analyze the results, including the top-ranked stocks, key metrics, and performance graphs.

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

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

4.  **Build with Docker (Optional):**
    ```bash
    # Build the container image
    docker build -t optiquant-app .

    # Run the container locally
    docker run -p 8501:8501 optiquant-app
    ```

---

### 📂 Project Structure


.
├── 📄 app.py                  # Main Streamlit application file
├── 📄 DataPreprocessing.py      # Feature engineering module
├── 📄 Dockerfile               # Instructions for building the Docker container
├── 📄 requirements.txt          # Python dependencies
├── 📦 model_lgbm.joblib        # Trained LightGBM model
├── 📦 model_cat.joblib         # Trained CatBoost model
└── 📦 model_rf.joblib           # Trained RandomForest model


---

### 🖼️ Screenshots

*(Placeholder: Insert screenshots of your app here to showcase the UI)*
**UI** 
<img width="2239" height="1248" alt="image" src="https://github.com/user-attachments/assets/25572489-43fa-4d1c-9495-877826bc63c7" />

**Performance Metrics & Graph:**
<img width="1174" height="550" alt="image" src="https://github.com/user-attachments/assets/0ead78c4-1139-4ab9-9199-d20d7238089f" />
