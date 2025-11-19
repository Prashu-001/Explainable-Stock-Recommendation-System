# 📈 Explainable Stock Recommendation System

AI-Powered Investment Intelligence for Personalized & Transparent Stock Selection

This project delivers an end-to-end machine-learning-driven stock recommendation system built using forecasting models, hybrid recommendation algorithms, and an interactive Streamlit dashboard. Designed to be explainable, robust, and production-ready, it analyzes 60+ NSE-listed stocks using fundamentals, technical indicators, user profiling, and risk-adjusted prediction models.

url - https://explainable-stock-recommendation-system.streamlit.app/

# 🚀 Features

<b>📊 Forecasting models</b>: ARIMA, ARIMA-GARCH, LSTM, LSTM-GARCH, and reliability-weighted ensembles

<b>🧩 Hybrid recommender system</b>: Content-based + Collaborative filtering + Risk profiling

🧠 Explainable ML with performance charts, radar plots, volatility-return visualizations

<b>🧍 User profiling</b>: risk appetite + sector preferences

<b>📉 Deep financial metrics</b>: volatility, Sharpe ratio, CAGR, profit margins, ROE/ROA etc.

🌐 Interactive Streamlit App to generate personalized recommendations

# 📂 Repository Structure
```
Explainable-Stock-Recommendation-System/
│
├── datasets/                         # Raw & processed stock datasets
│
├── app.py                            # Streamlit dashboard (main app)
├── load_and_preprocess_dataset.py    # Data loading, cleaning & feature engineering
├── models_training.py                # ARIMA, LSTM & Ensemble model training
├── models_evaluation.py              # Evaluation (RMSE, MAPE, CI, residuals)
├── statistical_tests.py              # Ljung-Box, ARCH, diagnostics
├── system_evaluation.py              # End-to-end recommender evaluation
├── visualization.py                  # Graphs & explainability visualizations
│
├── requirements.txt                  # All dependencies
├── README.md                         # Documentation
```

# 📊 Data Pipeline

1. Data Collection

60+ NSE stocks from sectors like IT, Banking, Pharma, Energy, FMCG

Live price + fundamentals fetched using Yahoo Finance API

2. Cleaning & Feature Engineering

Backfill for missing values

Daily returns, volatility, CAGR, mean returns

Sharpe ratio & advanced statistical transformations

Fundamentals integrated:

marketCap, ROE, ROA, profitMargins, governance risk scores

3. Encoding & Scaling

OneHotEncoding for sector/industry/region

MinMaxScaler for numerical features

👤 User Profiling
Risk Appetite (Real-world Investor Distribution)
Profile	% Users	Behavior
Conservative	40%	Prefers stable, low-volatility stocks
Balanced	40%	Medium-risk diversified picks
Aggressive	20%	High volatility & high growth
Sector Preferences

Each simulated user chooses 2 sectors → Helps generate realistic investment patterns.

Investment Probability Logic

Base probability: 0.2

Sector match bonus: +0.1

Risk-adjusted boost based on volatility & CAGR

Final matrix: 1000×58 synthetic user-stock investments

🤖 Forecasting Models
Models Used

ARIMA / ARIMA-GARCH for autocorrelation + volatility

LSTM / LSTM-GARCH for nonlinear temporal patterns

Ensemble Forecasting via reliability-weighted averaging

Model Selection Metric: Reliability Score
score = 0.3*(1/RMSE) +
        0.2*(1/MAPE) +
        0.4*(CI Coverage) +
        0.1*(Residual Std)


Best model per stock selected based on highest score.

🔍 Recommendation Engine Architecture
1. Content-Based Filtering

Similarity via fundamentals + performance metrics.

2. Collaborative Filtering

User–user patterns based on synthetic 1000×58 investment matrix.


🧮 Risk-Adjusted Ranking Algorithm

This system uses a custom scoring function to rank stocks based on a combination of:

Forecasted price direction (pred_signal)

Return distribution (lower_mean_returns, upper_mean_returns)

Volatility

User’s risk appetite

The ranking is generated in two steps:

1️⃣ Compute a Prediction Score

A base prediction score is calculated for every stock:

Pred_Score = 0.5 × Pred_Signal  
           + 0.25 × Lower_Mean_Returns  
           + 0.25 × Upper_Mean_Returns


This captures both return potential and prediction confidence.

2️⃣ Risk-Specific Adjustments

Different investors interpret volatility differently.
So we customize the score:

### 🟦 Conservative Investors

Prefer low-volatility, stable return stocks.

Adjusted_Score = (0.5×Pred_Signal 
                  + 0.35×Lower_Return 
                  + 0.15×Upper_Return)  
                  × (1 - Volatility)


Rewards stocks with lower volatility

Gives higher weight to downside protection

🟩 Balanced / Others

Prefer medium-risk opportunities.

Adjusted_Score = Base_Score × (1 - |Volatility - 0.5|)


Highest score when volatility ≈ 0.5

Penalizes very low or very high volatility

🟥 Aggressive Investors

Prefer high-volatility, high-growth opportunities.

Adjusted_Score = (0.5×Pred_Signal 
                  + 0.15×Lower_Return 
                  + 0.35×Upper_Return)  
                  × Volatility


Rewards high volatility + high upside returns

🎯 Final Ranking

After calculating the adjusted score, the system:

Sorts all stocks in descending order of pred_score

Returns top N ranked recommendations (default = 5)

This makes the system:

Personalized

Transparent

Fully explainable

Mathematically grounded

4. Hybrid Ranking

Final list = Merge candidates → Score → Top-k picks

❄️ Cold-Start Solutions
New Users

If user has no investment history → rank based on risk profile.

New Stocks

Use feature similarity (content-based filtering).

🖥️ Streamlit Interface (Explainable AI)

The dashboard includes:

✅ User Inputs

Select portfolio

Choose risk appetite

Generate recommendations

📈 Explainability Tabs

Performance (historical line chart)

Radar Chart (multi-metric comparison)

Feature Similarity Heatmap

Volatility vs CAGR Scatter

This gives users transparent reasoning behind every recommendation.

# ▶️ How to Run

1. Clone the repo
```
git clone https://github.com/Prashu-001/Explainable-Stock-Recommendation-System
cd Explainable-Stock-Recommendation-System
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Preprocess Dataset
```
python load_and_preprocess_dataset.py
```

4. Train Models
```
python models_training.py
```
5. Evaluate Models
```
python models_evaluation.py
```

6. Run Statistical Tests (optional)
```
python statistical_tests.py
```

8. System-Level Evaluation
```
python system_evaluation.py
```

9. Launch the Streamlit App
```
streamlit run app.py
```

# 📌 Future Enhancements


# 🙌 Acknowledgements

Yahoo Finance for data API

NSE for stock information

All open-source tools powering this project
