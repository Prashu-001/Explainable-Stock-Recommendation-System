import pickle
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import KNNImputer

np.random.seed(42)

tickers = [
    #------Tech/IT------
    "TCS.NS","INFY.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","WIPRO.NS","COFORGE.NS",
    "PERSISTENT.NS","MPHASIS.NS",

    # ---- Banking & Financial Services ----
    "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS",
    "BAJFINANCE.NS","HDFCAMC.NS","HDFCLIFE.NS","YESBANK.NS",

    # ---- Fintech / Digital ----
    "PAYTM.NS","POLICYBZR.NS",

    # ---- Pharma & Healthcare ----
    "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","LUPIN.NS","DIVISLAB.NS",
    "APOLLOHOSP.NS","FORTIS.NS",

    # ---- Energy / Oil & Gas ----
    "RELIANCE.NS","ONGC.NS","POWERGRID.NS","IOC.NS","GAIL.NS","BPCL.NS",
    # ---- Utilities (Renewables, Power) ----
    "NTPC.NS","ADANIGREEN.NS","TATAPOWER.NS",
    # ---- Consumer Defensive / FMCG ----
    "HINDUNILVR.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS","GODREJCP.NS",
    "ITC.NS","MARICO.NS",
    # ---- Consumer Cyclical / Auto ----
    "MARUTI.NS","TATAMOTORS.NS","HEROMOTOCO.NS","EICHERMOT.NS","BOSCHLTD.NS",
    "BAJAJ-AUTO.NS",
    # ---- Metals & Mining ----
    "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS",
    # ---- Telecom ----
    "BHARTIARTL.NS","IDEA.NS",
    # ---- Infrastructure / Industrials ----
    "LT.NS","ADANIENT.NS","ULTRACEMCO.NS","SHREECEM.NS"
]

years = 3
df = yf.download(tickers, period = '3y', interval = '1d')['Close']
df.columns = ['Close_' + col.split('.')[0] for col in df.columns]
null=df.isnull().sum()
cols_with_missings=list(null[null>0].index)
df[cols_with_missings]=df[cols_with_missings].bfill()

info_dict = {}
for ticker in tickers:
    col = 'Close_' + ticker.split('.')[0]
    returns_col = 'returns_' + ticker.split('.')[0]

    # Calculate daily returns
    df[returns_col] = df[col].pct_change()

    # Drop NaN for calculations
    valid_returns = df[returns_col].dropna()

    # Calculate time span in years
    years = (df.index[-1] - df.index[0]).days / 365

    # Calculate metrics
    cagr = ((df[col].iloc[-1] / df[col].iloc[0]) ** (1/years)) - 1
    volatility = valid_returns.std()
    mean_return = valid_returns.mean()
    sharpe_ratio = mean_return / volatility if volatility != 0 else 0

    # Store results
    info_dict[ticker] = {
        "CAGR": cagr,
        "Volatility": volatility,
        "Mean_Return": mean_return,
        "Sharpe_Ratio": sharpe_ratio
    }

# Convert to DataFrame for better readability
metrics_df = pd.DataFrame(info_dict).T

with open("datasets/models_data.pkl", "rb") as f:
    models_data = pickle.load(f)

models_data['symbol'] = models_data['symbol'] + '.NS'
models_data = models_data.set_index('symbol')

metrics_df['pred_signal'] = models_data['returns'].apply(lambda x: np.mean(np.array(x)-1) if isinstance(x, list) else np.nan)
metrics_df['mean_returns'] = models_data['returns'].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
metrics_df['lower_mean_returns'] = models_data['lower'].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
metrics_df['upper_mean_returns'] = models_data['upper'].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
metrics_df.reset_index(names='symbol', inplace = True)
metrics_df.to_csv('datasets/metrics_data.csv')
metrics_df.set_index('symbol',inplace = True)
# stocks content.
data = []
for ticker in tickers:
    info = yf.Ticker(ticker).info
    
    # Categorical features
    sector = info.get('sector', None)
    industry = info.get('industry', None)
    region = info.get('region', None)
    
    # Numerical features
    features = {
        'marketCap': info.get('marketCap', np.nan),
        'beta': info.get('beta', np.nan),
        'returnOnEquity': info.get('returnOnEquity', np.nan),
        'returnOnAssets': info.get('returnOnAssets', np.nan),
        'profitMargins': info.get('profitMargins', np.nan),
        'grossMargins': info.get('grossMargins', np.nan),
        'ebitdaMargins': info.get('ebitdaMargins', np.nan),
        'debtToEquity': info.get('debtToEquity', np.nan),
        'auditRisk': info.get('auditRisk', np.nan),
        'boardRisk': info.get('boardRisk', np.nan),
        'compensationRisk': info.get('compensationRisk', np.nan),
        'shareHolderRightsRisk': info.get('shareHolderRightsRisk', np.nan),
        'overallRisk': info.get('overallRisk', np.nan)
    }
    
    data.append({
        'symbol': ticker,
        'sector': sector,
        'industry': industry,
        'region': region,
        **features
    })

content_df = pd.DataFrame(data)
content_df.set_index('symbol',inplace=True)
content_df = pd.concat([content_df,metrics_df],axis=1)

stocks = {}
for symbol in content_df.index:
    stocks[symbol] = content_df['sector'][symbol]

# Handle categorical features with One-Hot Encoding
categorical_cols = ['sector', 'industry', 'region']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cat = encoder.fit_transform(content_df[categorical_cols])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_cols))

# 3. Handle numerical features - fill NaN and scale
numerical_cols = ['marketCap', 'beta', 'returnOnEquity', 'returnOnAssets','profitMargins', 'grossMargins', 
                  'ebitdaMargins', 'debtToEquity','auditRisk', 'boardRisk', 'compensationRisk', 
                  'shareHolderRightsRisk', 'overallRisk','CAGR','Volatility','Mean_Return','Sharpe_Ratio']

num_df = content_df[numerical_cols].fillna(content_df[numerical_cols].median())
scaler = MinMaxScaler()
scaled_num = scaler.fit_transform(num_df)
scaled_num_df = pd.DataFrame(scaled_num, columns=numerical_cols)

clean_df = pd.concat([scaled_num_df.reset_index(drop=True), encoded_cat_df.reset_index(drop=True)],axis=1)
clean_df.to_csv('datasets/stocks_content_data.csv')
scaled_num_df['symbol'] = metrics_df.index
scaled_num_df['sector'] = scaled_num_df['symbol'].map(stocks)

# -----------------------------
# Create user profiles
# -----------------------------
risk_profiles = ["Conservative", "Balanced", "Aggressive"]
sector_preferences = list(set(stocks.values()))
num_users = 100

users = []
for i in range(1, num_users+1):
    sector1 = np.random.choice(sector_preferences)
    sector2 = np.random.choice(sector_preferences)
    users.append({
        "user_id": f"U{i}",
        "risk": np.random.choice(risk_profiles, p=[0.4, 0.4, 0.2]),
        "sector": [sector1, sector2]
    })
users_df = pd.DataFrame(users)

# -----------------------------
# Investment probability function
# -----------------------------
def get_investment_prob(stock_metrics, user_risk, user_sector):
    prob = 0.2  # base probability

    # Sector preference
    if stock_metrics['sector'] in user_sector:
        prob += 0.3

    # Risk-based adjustments
    if user_risk == "Conservative":
        prob += (1 - stock_metrics['Volatility']) * 0.3  # low volatility boost
        prob += stock_metrics['Mean_Return'] * 0.1
    elif user_risk == "Balanced":
        prob += (1 - abs(stock_metrics['Volatility'] - 0.5)) * 0.2
    elif user_risk == "Aggressive":
        prob += stock_metrics['Volatility'] * 0.3
        prob += stock_metrics['CAGR'] * 0.2

    return min(max(prob, 0), 1)

# -----------------------------
# Simulate investments with top-k stocks
# -----------------------------
def simulate_investments(user_risk, user_sector):
    # Calculate probability for each stock
    probs = scaled_num_df.apply(lambda row: get_investment_prob(row, user_risk, user_sector), axis=1)
    
    # Randomly pick 5-10 stocks to invest in
    k = np.random.randint(3, 11)
    top_k_indices = probs.nlargest(k).index
    
    # Initialize all stocks as 0
    investments = {stock: 0 for stock in scaled_num_df['symbol']}
    
    # Mark top-k stocks as invested
    for idx in top_k_indices:
        investments[scaled_num_df.loc[idx, 'symbol']] = 1
    
    return investments

# Generate investment matrix
investment_data = []
for _, row in users_df.iterrows():
    inv = simulate_investments(row['risk'], row['sector'])
    inv['user_id'] = row['user_id']
    investment_data.append(inv)

investments_df = pd.DataFrame(investment_data).set_index('user_id')
# ----------------------------
# Ensure every stock has at least one investor
# ----------------------------
for stock in scaled_num_df['symbol']:
    if investments_df[stock].sum() == 0:
        random_user = np.random.choice(investments_df.index)
        investments_df.at[random_user, stock] = 1

investments_df.to_csv('datasets/users_investment_data.csv')