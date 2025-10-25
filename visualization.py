import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import pi


def plot_series(series: pd.Series, title: str = "Series"):
    plt.figure(figsize=(10, 3))
    plt.plot(series.index, series.values, label=title)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_forecast(train: pd.Series, test: pd.Series, preds: np.ndarray, lower = None, upper = None, title = "Forecast"):
    plt.figure(figsize=(10, 4))
    plt.plot(test.index, test.values, label='True', color='black')
    plt.plot(test.index, preds, label='Predicted', color='red', linestyle='--')
    if lower is not None and upper is not None:
        plt.fill_between(test.index, lower, upper, color='gray', alpha=0.25, label='CI')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_stock_radar(symbols, df):
    features = ['CAGR', 'Volatility', 'Sharpe_Ratio', 'Mean_Return']
    N = len(features)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    for sym in symbols:
        values = df.loc[sym, features].values.flatten().tolist()
        values += values[:1]
        plt.polar(angles, values, label=sym)
    plt.xticks(angles[:-1], features)
    plt.title("Stock Feature Profiles")
    plt.legend(loc='upper right')
    plt.show()

def plot_feature_similarity(target, recs, df, features):
    target_vec = df.loc[target, features]
    diffs = {}
    for r in recs:
        diffs[r] = abs(target_vec - df.loc[r, features])
    diff_df = pd.DataFrame(diffs)
    diff_df.plot(kind='bar', figsize=(8,4), title=f"Feature Distance from {target}")
    plt.ylabel("Absolute Difference")
    plt.show()

def plot_volatility_cagr(metrics_df: pd.DataFrame, recommend, taken_stocks):
    plt.figure(figsize=(6,4))
    plt.scatter(metrics_df['Volatility'], metrics_df['CAGR'], alpha=0.5, label="All Stocks")
    plt.scatter(metrics_df.loc[recommend, 'Volatility'], metrics_df.loc[recommend, 'CAGR'], color='red', label='Recommended')
    plt.scatter(metrics_df.loc[taken_stocks, 'Volatility'], metrics_df.loc[taken_stocks, 'CAGR'], color='gold', s=100, label='TCS.NS')
    plt.xlabel("Volatility")
    plt.ylabel("CAGR")
    plt.legend()
    plt.title("Risk vs Return Space with Recommendations Highlighted")
    plt.show()
