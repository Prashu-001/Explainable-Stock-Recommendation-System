import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import pi

# -----------------------------
# Time series plot
# -----------------------------
def plot_series(series: pd.Series, title: str = "Series"):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(series.index, series.values, label=title, color='tab:blue')
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()

    explanation = (
        f"**{title}** shows how the stock value changes over time. "
        "Trends going upward indicate growth, while downward movements may indicate declining performance. "
        "You can use this to observe general patterns or volatility in the selected stock or feature."
    )

    return fig, explanation


# -----------------------------
# Forecast plot with confidence interval
# -----------------------------
def plot_forecast(test: pd.Series, preds: np.ndarray, lower=None, upper=None, title: str = "Forecast"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test.index, test.values, label='True', color='black')
    ax.plot(test.index, preds, label='Predicted', color='red', linestyle='--')
    if lower is not None and upper is not None:
        ax.fill_between(test.index, lower, upper, color='gray', alpha=0.25, label='Confidence Interval')
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Value")
    ax.legend()
    fig.tight_layout()

    explanation = (
        f"**{title}** compares model predictions (red dashed line) with the actual values (black line). "
        "If the predicted and actual lines overlap closely, the model performs well. "
        "The shaded region (if shown) represents uncertainty — wider intervals indicate more uncertainty."
    )

    return fig, explanation


# -----------------------------
# Stock radar chart
# -----------------------------
def plot_stock_radar(symbols, df):
    features = ['CAGR', 'Volatility', 'Sharpe_Ratio', 'Mean_Return']
    N = len(features)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for sym in symbols:
        values = df.loc[sym, features].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, label=sym)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_title("Stock Feature Profiles")
    ax.legend(loc='upper right')

    explanation = (
        "This **radar chart** compares multiple stocks across four performance metrics: "
        "**CAGR (growth rate)**, **Volatility (risk)**, **Sharpe Ratio (risk-adjusted return)**, "
        "and **Mean Return**. Each line represents a stock — a larger and more balanced area "
        "generally indicates better overall performance with lower risk."
    )

    return fig, explanation


# -----------------------------
# Feature similarity bar plot
# -----------------------------
def plot_feature_similarity(target, recs, df, features):
    target_vec = df.loc[target, features]
    diffs = {}
    for r in recs:
        diffs[r] = abs(target_vec - df.loc[r, features])
    
    diff_df = pd.DataFrame(diffs)
    fig, ax = plt.subplots(figsize=(8, 4))
    diff_df.plot(kind='bar', ax=ax, title=f"Feature Distance from {target}")
    ax.set_ylabel("Absolute Difference")
    fig.tight_layout()

    explanation = (
        f"This bar chart shows how **recommended stocks differ from {target}** across the selected features. "
        "Shorter bars mean the stock behaves similarly to the target stock, while taller bars indicate larger deviations. "
        "It helps identify which stocks have similar risk-return characteristics."
    )

    return fig, explanation


# -----------------------------
# Volatility vs CAGR scatter plot
# -----------------------------
def plot_volatility_cagr(metrics_df: pd.DataFrame, recommend, taken_stocks):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(metrics_df['Volatility'], metrics_df['CAGR'], alpha=0.5, label="All Stocks")
    ax.scatter(metrics_df.loc[recommend, 'Volatility'], metrics_df.loc[recommend, 'CAGR'],
               color='red', label='Recommended')
    ax.scatter(metrics_df.loc[taken_stocks, 'Volatility'], metrics_df.loc[taken_stocks, 'CAGR'],
               color='gold', s=100, label='Taken by You')
    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("CAGR (Return)")
    ax.set_title("Risk vs Return Space with Recommendations Highlighted")
    ax.legend()
    fig.tight_layout()

    explanation = (
        "This **scatter plot** visualizes the classic **Risk vs Return tradeoff**. "
        "Stocks on the upper-left region (high CAGR, low Volatility) are ideal — high return with low risk. "
        "Your selected stocks are shown in gold, and the system’s recommendations in red. "
        "This helps you compare where your choices lie in the risk-return space."
    )

    return fig, explanation