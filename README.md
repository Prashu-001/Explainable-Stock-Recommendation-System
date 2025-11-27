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
|-- Final Project Report
|-- final ppt
├── README.md                         # Documentation
```


## How to Run

1. Clone the repo
```
git clone https://github.com/Prashu-001/Explainable-Stock-Recommendation-System
cd Explainable-Stock-Recommendation-System
```

2. Install dependencies
```
pip install -r requirements.txt
```
3. Evaluate Models
```
python models_evaluation.py
```
4. Statistical Tests
```
python statistical_tests.py
```
5. Train Models
```
python models_training.py
```
6. Preprocess Dataset
```
python load_and_preprocess_dataset.py
```
7. Launch the Streamlit App
```
streamlit run app.py
```

# 🙌 Acknowledgements

Yahoo Finance for data API

NSE for stock information

All open-source tools powering this project
