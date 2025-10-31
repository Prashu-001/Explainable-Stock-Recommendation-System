import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from visualization import plot_series, plot_forecast, plot_stock_radar, plot_feature_similarity, plot_volatility_cagr
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Load Data
# -----------------------------
#@st.cache_data
def load_data():
    clean_df = pd.read_csv('datasets/stocks_content_data.csv')
    metrics_df = pd.read_csv('datasets/metrics_data.csv')
    metrics_df = metrics_df.set_index('symbol')
    investments_df = pd.read_csv('datasets/users_investment_data.csv')
    investments_df = investments_df.set_index('user_id')

    with open("datasets/models_data.pkl", "rb") as f:
        models_data = pickle.load(f)

    models_data['symbol'] = models_data['symbol'] + '.NS'
    models_data = models_data.set_index('symbol')
    
    # Create content-based similarity
    features = clean_df
    similarity_matrix = cosine_similarity(features)
    similarity_df = pd.DataFrame(similarity_matrix, index=metrics_df.index, columns=metrics_df.index)
    
    # Collaborative filtering item-item similarity
    item_similarity = cosine_similarity(investments_df.T)
    item_sim_df = pd.DataFrame(item_similarity, index=investments_df.columns, columns=investments_df.columns)
    
    return metrics_df, investments_df, similarity_df, item_sim_df, models_data

metrics_df, investments_df, similarity_df, item_sim_df, models_data = load_data()
# -----------------------------
# Content similarity
# -----------------------------

def recommend_similar(stock_symbol, top_n=10):
    if stock_symbol not in similarity_df.index:
        return []
    similar_scores = similarity_df[stock_symbol].sort_values(ascending=False)
    similar_stocks = similar_scores.iloc[1:top_n+1].index.tolist()  # exclude itself
    return similar_stocks

def get_similar_stocks(user_stocks, top_n=10):
    similar_set = set()
    for stock in user_stocks:
        similar_set.update(recommend_similar(stock, top_n=top_n))
    similar_set.difference_update(user_stocks)  # remove already held stocks
    return list(similar_set)

# -----------------------------
# Collaborative filtering
# -----------------------------

def recommend_cf(user_stocks, top_n=10):
    if not user_stocks:
        return []
    
    # Create user preference vector (1 for owned stocks)
    user_vector = np.zeros(item_sim_df.shape[0])
    for s in user_stocks:
        idx = item_sim_df.index.get_loc(s)
        user_vector[idx] = 1  # or you can assign portfolio weights if available
    
    # Weighted sum of item similarities (like CF)
    scores = item_sim_df.values.dot(user_vector)
    scores = pd.Series(scores, index=item_sim_df.index)
    
    # Remove already owned stocks
    scores = scores.drop(user_stocks, errors="ignore")
    
    # Return top N recommended stocks
    return scores.sort_values(ascending=False).head(top_n).index.tolist()


# -----------------------------
# Rank by predicted returns + risk
# -----------------------------
def rank_stocks_by_prediction_and_risk(stock_list, user_risk, top_final=5):
    if stock_list == 'None':
        subset = metrics_df.copy()
    else:
        subset = metrics_df.loc[stock_list].copy()
    
    # Base score
    subset['pred_score'] = (0.5*subset['pred_signal'] +
                            0.25*subset['lower_mean_returns'] +
                            0.25*subset['upper_mean_returns'])
    
    # Risk adjustment
    if user_risk == "Conservative":
        # Give more weights to the stock with higher lower returns and lower volatility.
        subset['pred_score'] = (0.5*subset['pred_signal'] +
                                0.35*subset['lower_mean_returns'] +
                                0.15*subset['upper_mean_returns'])
        subset['pred_score'] *= (1 - subset['Volatility'])
    elif (user_risk == "Balanced") or (user_risk == "Others"):
        subset['pred_score'] *= (1 - abs(subset['Volatility'] - 0.5))
    elif user_risk == "Aggressive":
        # Give more weights to the stocks with higher upper returns and higher volatility.
        subset['pred_score'] = (0.5*subset['pred_signal'] +
                                0.15*subset['lower_mean_returns'] +
                                0.35*subset['upper_mean_returns'])
        subset['pred_score'] *= subset['Volatility']
    
    return subset.sort_values('pred_score', ascending=False).index.tolist()[:top_final]

# -----------------------------
# Hybrid recommendation pipeline
# -----------------------------
def hybrid_recommendation(user_stocks, user_risk, top_similar=10, top_cf=10, top_final=5):
    # 1. Content-based candidates
    content_candidates = get_similar_stocks(user_stocks, top_n=top_similar)  
    # 2. Collaborative filtering candidates
    cf_candidates = recommend_cf(user_stocks, top_n=top_cf)
    # 3. Merge candidates
    all_candidates = list(set(content_candidates + cf_candidates))
    # 4. Rank by predicted return + risk
    ranked_candidates = rank_stocks_by_prediction_and_risk(all_candidates, user_risk, top_final)
    
    # 5. Return top-N recommendations
    return ranked_candidates

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="AI Stock Recommender", page_icon="📈", layout="wide")

# -----------------------------
# Custom CSS for Styling
# -----------------------------
st.markdown("""
    <style>
        /* Background and Font */
        body {
            background-color: #f7f9fc;
            color: #2c3e50;
            font-family: 'Inter', sans-serif;
        }
        .main-title {
            font-size: 2.2em;
            font-weight: 700;
            color: white;
            background: linear-gradient(90deg, #2c3e50, #3498db);
            padding: 1rem 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .section {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            height: 3em;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            color: white;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.03);
            background: linear-gradient(90deg, #2ecc71, #3498db);
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown('<div class="main-title">📈 AI-Powered Stock Recommendation System</div>', unsafe_allow_html=True)
    st.markdown("""
    Get **personalized stock recommendations** based on your:
    - 📊 Current portfolio  
    - ⚖️ Risk profile  
    - 🚀 Predicted performance and volatility metrics  
    """)
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2331/2331953.png", width=180)

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------
st.markdown("### 🧩 Customize Your Profile")

col1, col2 = st.columns([2, 1])
with col1:
    all_stocks = metrics_df.index.tolist() + ['None']
    user_stocks = st.multiselect("Select Stocks in Your Portfolio:", options=all_stocks)
    user_stocks = [s for s in user_stocks if s != "None"]
with col2:
    risk_profile = st.radio("Select Your Risk Profile:", ["Conservative", "Balanced", "Aggressive", "Others"])

st.markdown("---")

# -----------------------------
# Recommendation Section
# -----------------------------
if st.button("Get My Recommendations"):
    if not user_stocks:
        recs = rank_stocks_by_prediction_and_risk('None', risk_profile, top_final=5)
    else:
        recs = hybrid_recommendation(user_stocks, risk_profile, top_similar=10, top_final=5)

    st.markdown("## 🎯 Top Recommended Stocks")
    rec_df = metrics_df.loc[recs, ['mean_returns', 'lower_mean_returns', 'upper_mean_returns', 'Volatility']]
    st.dataframe(rec_df.style.highlight_max(axis=0, color="#2AA456"))

    st.markdown("## 📊 Deep Dive Analytics")

    for stock in recs:
        with st.expander(f"🔍 Detailed Insights for {stock}", expanded=False):
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🧭 Radar", "⚙️ Feature Similarity", "📉 Volatility vs CAGR"])

            with tab1:
                series = yf.download(stock, period='3y', interval='1d')['Close']
                fig, desc = plot_series(series, title=f"{stock} Price Trend")
                st.pyplot(fig)
                st.markdown(desc)

            with tab2:
                fig, desc = plot_stock_radar([stock] + recs, metrics_df)
                st.pyplot(fig)
                st.markdown(desc)

            with tab3:
                features = ['CAGR', 'Volatility', 'Sharpe_Ratio', 'Mean_Return']
                fig, desc = plot_feature_similarity(stock, recs, metrics_df, features)
                st.pyplot(fig)
                st.markdown(desc)

            with tab4:
                fig, desc = plot_volatility_cagr(metrics_df, stock, user_stocks)
                st.pyplot(fig)
                st.markdown(desc)

st.markdown("--")
st.caption("💡 Developed by **Prashu Poras | IIT Guwahati** · Powered by AI · © 2025")
