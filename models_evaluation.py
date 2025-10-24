import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error as mse, mean_absolute_percentage_error

def compute_metrics(true, pred):
    rmse = math.sqrt(mse(true,pred))
    mape = mean_absolute_percentage_error(true,pred) if true.sum()!=0 else np.nan
    residuals = true - pred
    resid_std = np.std(residuals)
    return {'rmse':rmse,'mape':mape,'resid_std':resid_std}

def compute_ci_coverage(true, lower, upper):
    contained = ((true>=lower) & (true<=upper)).sum()
    return float(contained/len(true))

def reliability_score(metrics, ci_coverage: float, weights=None):
    if weights==None:
        weights = {'rmse':0.3,'mape':0.2,'ci':0.4,'resid':0.1}
    score = 0.0
    score += weights['rmse']*(1.0/(metrics['rmse']+1e-8))
    score += weights['mape']*(1.0/(metrics['mape']+1e-8))
    score += weights['ci']*ci_coverage
    score += weights['resid']*metrics['resid_std']
    return float(score)
