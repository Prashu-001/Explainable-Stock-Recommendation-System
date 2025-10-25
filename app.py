import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

    
    # Create content-based similarity
    features = clean_df
    similarity_matrix = cosine_similarity(features)
    similarity_df = pd.DataFrame(similarity_matrix, index=metrics_df.index, columns=metrics_df.index)
    
    # Collaborative filtering item-item similarity
    item_similarity = cosine_similarity(investments_df.T)
    item_sim_df = pd.DataFrame(item_similarity, index=investments_df.columns, columns=investments_df.columns)
    
    return metrics_df, investments_df, similarity_df, item_sim_df

metrics_df, investments_df, similarity_df, item_sim_df = load_data()
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
def rank_stocks_by_prediction_and_risk(stock_list, user_risk):
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
    elif user_risk == "Balanced":
        subset['pred_score'] *= (1 - abs(subset['Volatility'] - 0.5))
    elif user_risk == "Aggressive":
        # Give more weights to the stocks with higher upper returns and higher volatility.
        subset['pred_score'] = (0.5*subset['pred_signal'] +
                                0.15*subset['lower_mean_returns'] +
                                0.35*subset['upper_mean_returns'])
        subset['pred_score'] *= subset['Volatility']
    
    return subset.sort_values('pred_score', ascending=False).index.tolist()

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
    ranked_candidates = rank_stocks_by_prediction_and_risk(all_candidates, user_risk)
    
    # 5. Return top-N recommendations
    return ranked_candidates[:top_final]


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Stock Recommender", page_icon="📈", layout="wide")
st.title("📈 AI-Powered Stock Recommendation System")

st.markdown("""
This app recommends stocks based on:
- Your current portfolio  
- Your risk profile  
- Predicted performance and volatility metrics
""")

# -----------------------------
# User Inputs
# -----------------------------
all_stocks = metrics_df.index.tolist()
user_stocks = st.multiselect("Select Stocks in Your Portfolio:", options=all_stocks)

risk_profile = st.radio("Select Your Risk Profile:", ["Conservative", "Balanced", "Aggressive"])

if st.button("Get Recommendations"):
    if not user_stocks:
        st.warning("⚠️ Please select at least one stock.")
    else:
        recs = hybrid_recommendation(user_stocks, risk_profile, top_similar=10, top_final=5)
        st.subheader("🎯 Hybrid Stock Recommendations")
        rec_df = metrics_df.loc[recs, ['pred_signal', 'lower_mean_returns', 'upper_mean_returns', 'Volatility']]
        st.dataframe(rec_df)

st.markdown("---")
st.caption("Developed by Prashu Poras | IITG")