from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.seasonal import STL
from scipy.stats import linregress
import pandas as pd
import numpy as np

def adf_test(series: pd.Series):
    res = adfuller(series.dropna(), autolag = 'AIC')
    return {'adf_stat':res[0],'pvalue':res[1],'usedlag':res[2],'nobs':res[3]}

def engle_arch_test(series: pd.Series,lags: int = 5):
    res = het_arch(series.dropna(),nlags = lags)
    return {'lm_stat':res[0],'lm_pvalue':res[1],'f_stat':res[2],'f_pvalue':res[3]}

def ljung_box_test(series: pd.Series, lags = [10]):
    res = acorr_ljungbox(series.dropna(),lags = lags, return_df = True)
    return {'lb_stat':res['lb_stat'].iloc[-1],'lb_pvalue':res['lb_pvalue'].iloc[-1]}

def stl_decompose(series: pd.Series,period: int):
    stl = STL(series.dropna(),period = period,robust=True)
    res = stl.fit()
    return {'trend': res.trend, 'seasonal': res.seasonal, 'resid': res.resid}

def trend_strength(series: pd.Series):
    y = series.dropna().values
    x = np.arange(len(y))
    if len(y)< 3:
        return 0.0
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return float(abs(slope))