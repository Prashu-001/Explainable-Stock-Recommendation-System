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

def evaluate_recommender(test_users, K=5):
    precision_list, recall_list, hit_list = [], [], []

    for user in test_users:
        true_stocks = set(investments_df.loc[user][investments_df.loc[user] == 1].index)
        if not true_stocks:
            continue
        
        # Recommend top K stocks
        recs = hybrid_recommendation(list(true_stocks), 
                                     user_risk="Balanced", 
                                     top_similar=10, 
                                     top_final=K)

        recs = set(recs)

        # Intersection
        hits = true_stocks.intersection(recs)
        hit_rate = 1 if len(hits) > 0 else 0

        precision = len(hits) / K
        recall = len(hits) / len(true_stocks)

        precision_list.append(precision)
        recall_list.append(recall)
        hit_list.append(hit_rate)

    return {
        "Precision@K": np.mean(precision_list),
        "Recall@K": np.mean(recall_list),
        "HitRate": np.mean(hit_list)
    }

test_users = investments_df.sample(50).index
eval_result = evaluate_recommender(test_users)
st.write("📊 Recommendation Evaluation:", eval_result)