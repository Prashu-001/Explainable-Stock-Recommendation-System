import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import pmdarima as pm
from arch import arch_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from sklearn.metrics import mean_squared_error as mse, mean_absolute_percentage_error
from itertools import product

from statistical_tests import ljung_box_test, engle_arch_test
from models_evaluation import compute_metrics, compute_ci_coverage, reliability_score

def train_arima(series: pd.Series, seasonal: bool = False, m: int = 1):
    model = pm.auto_arima(series, seasonal=seasonal, m=m, error_action='ignore', suppress_warnings=True)
    return {'model': model, 'name': 'ARIMA', 'order': model.order, 'seasonal_order': model.seasonal_order}

def forecast_arima(model_obj, steps: int = 5):
    return model_obj.predict(n_periods=steps)

def build_lstm_functional(input_shape, lstm_units=50, dropout_rate=0.2, dense_units=1):
    inp = Input(shape=input_shape)
    x = LSTM(lstm_units)(inp)
    x = Dropout(dropout_rate)(x)
    out = Dense(dense_units)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

def tune_lstm(x_train, y_train,x_test, y_test, lstm_units_list=[64,86,128], dropout_list=[0.1], epochs=80, batch_size=16):
    best_model = None
    best_score = np.inf
    best_params = {}
    
    for lstm_units, dropout_rate in product(lstm_units_list, dropout_list):
        #x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        model = build_lstm_functional((x_train.shape[1],1), lstm_units=lstm_units, dropout_rate=dropout_rate)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        preds = model.predict(x_test, verbose=0).flatten()
        rmse = np.sqrt(mse(y_test, preds))
        
        if rmse < best_score:
            best_score = rmse
            best_model = model
            best_params = {'lstm_units': lstm_units, 'dropout_rate': dropout_rate}
            residuals = y_test - preds
    
    return {'model': best_model, 'params': best_params, 'preds': preds, 'rmse':best_score, 'residuals': residuals}

def forecast_lstm(model_obj, series: pd.Series, steps: int = 5) -> np.ndarray:
    seq = list(series.values[-lookback:])
    preds = []
    for _ in range(steps):
        x = np.array(seq[-lookback:]).reshape((1, lookback, 1))
        p = float(model_obj.predict(x, verbose=0)[0][0])
        preds.append(p)
        seq.append(p)
    return np.array(preds)

def train_garch(returns: pd.Series, p: int = 1, q: int = 1, mean: str = 'Zero', vol: str = 'Garch', dist: str = 'normal'):
    # returns should be a stationary series (log-returns or resid in returns)
    am = arch_model(returns, mean=mean, vol=vol, p=p, q=q, dist=dist)
    res = am.fit(disp='off')
    return {'model': res, 'name': f'GARCH({p},{q})'}

def forecast_garch(res, horizon: int = 5):
    fc = res.forecast(horizon=horizon, reindex=False)
    # variance forecasts for next horizon steps
    var_fc = fc.variance.iloc[-1].values  # shape (horizon,)
    std_fc = np.sqrt(var_fc)
    # sometimes fc.mean exists, otherwise zeros
    mean_fc = None
    try:
        mean_fc = fc.mean.iloc[-1].values
    except Exception:
        mean_fc = np.zeros_like(std_fc)
    return {'variance': var_fc, 'std': std_fc, 'mean': mean_fc}

def preds_to_returns(preds: np.ndarray, residuals):
    returns = np.exp(preds).cumprod()
    
    # naive CI using in-sample resid std (heuristic)
    resid_std = np.std(residuals) if residuals is not None else 0.0
    lower = returns - 1.96 * resid_std
    upper = returns + 1.96 * resid_std
    return returns, lower, upper

def build_arima(arima_train, n_test, arima_test):
    arima_res = train_arima(arima_train)
    arima_preds = forecast_arima(arima_res['model'], steps=n_test)
    # compute in-sample residuals for ARIMA
    arima_in_sample_pred = arima_res['model'].predict_in_sample()
    arima_residuals = arima_train.values[-len(arima_in_sample_pred):] - arima_in_sample_pred
    returns, lower, upper = preds_to_returns( np.array(arima_preds), arima_residuals)
    
    model = 'ARIMA'
    # Residual diagnostics
    idx = arima_train.index[-len(arima_residuals):]
    res_series = pd.Series(arima_residuals, index=idx)
    arima_ljungbox = ljung_box_test(res_series, lags=[4])
    arima_arch = engle_arch_test(res_series, lags=4)

    # If ARCH effect present -> fit GARCH on residuals.
    if arima_arch['lm_pvalue'] < 0.05:
        model = 'ARIMA-GARCH'
        garch_res = train_garch(res_series, p=1, q=1)
        garch_fc = forecast_garch(garch_res['model'], horizon=n_test)
        # combine ARIMA mean (in model-space) + GARCH std to form probabilistic forecasts (model-space)
        stds = garch_fc['std']  # std of returns for each horizon
        mean_preds = np.array(arima_preds)
        # cumulative variances
        cum_var = np.cumsum(garch_fc['variance'])
        lower = np.exp(np.cumsum(mean_preds) - 1.96 * np.sqrt(cum_var))
        upper = np.exp(np.cumsum(mean_preds) + 1.96 * np.sqrt(cum_var))
        #lower = last_price * np.exp(mean_preds - 1.96 * garch_fc['variance'])
        #upper = last_price * np.exp(mean_preds + 1.96 * garch_fc['variance'])
        
    # evaluate against true prices
    metrics = compute_metrics(arima_test.values, arima_preds)
    ci_cov = compute_ci_coverage(np.exp(arima_test.values).cumprod(), lower, upper)
    score = reliability_score(metrics, ci_cov)
    if model == 'ARIMA':
        models = arima_res['model']
    else:
        models = {'arima': arima_res['model'], 'garch': garch_res['model']}
        
    return {'model_name':model, 'meta': models, 'preds':arima_preds, 'returns': returns.tolist(),'lower': lower.tolist(), 
                            'upper': upper.tolist(),'metrics': metrics, 'ci_cov': ci_cov, 'score': score}, arima_ljungbox, arima_arch

def build_lstm(x_train, y_train,x_test, y_test):
     # Train LSTM on series_model (prefer stationary series)
    lstm_res = tune_lstm(x_train, y_train,x_test, y_test)
    # convert to price if needed
    lstm_returns, lower, upper = preds_to_returns(np.array(lstm_res['preds']), lstm_res['residuals'])
    model = 'LSTM'
    
    in_sample_preds = lstm_res['model'].predict(x_train, verbose=0).flatten()
    in_sample_y = y_train.values
    training_residuals = in_sample_y - in_sample_preds
    rs = pd.Series(training_residuals, index=x_train.index)
    lstm_ljungbox = ljung_box_test(rs, lags=[4])
    lstm_arch = engle_arch_test(rs, lags=4)

    # If ARCH effect present -> fit GARCH on residuals.
    if lstm_arch['lm_pvalue'] < 0.05:
        model = 'LSTM-GARCH'
        n_test = x_test.shape[0]
        garch_res = train_garch(rs, p=1, q=1)
        garch_fc = forecast_garch(garch_res['model'], horizon=n_test)
        mean_preds = np.array(lstm_res['preds'])
        
        # cumulative variances
        cum_var = np.cumsum(garch_fc['variance'])
        lower = np.round(np.exp(np.cumsum(mean_preds) - 1.96 * np.sqrt(cum_var)),5)
        upper = np.round(np.exp(np.cumsum(mean_preds) + 1.96 * np.sqrt(cum_var)),5)

    # evaluate against true prices
    metrics = compute_metrics(y_test.values, lstm_res['preds'])
    ci_cov = compute_ci_coverage(np.exp(y_test.values).cumprod(), lower, upper)
    score = reliability_score(metrics, ci_cov)
    if model == 'LSTM':
        models = lstm_res['model']
    else:
        models = {'lstm': lstm_res['model'], 'garch': garch_res['model']}
    
    return {'model_name':model, 'meta': models, 'returns': lstm_returns.tolist(), 'preds':lstm_res['preds'],'lower': lower.tolist(), 'upper': upper.tolist(),
                                                'metrics': metrics, 'ci_cov': ci_cov, 'score': score}, lstm_ljungbox, lstm_arch

                  
def pipeline_for_stock(symbol, df, forecast_horizon=10, stl_period=5, return_period=5):
    col = symbol
    symbol = symbol.split('_')[1]
    out = {'symbol': symbol, 'tests': {}, 'models': {}}

    series = df[col].ffill().dropna()
    y = np.log(series).diff(5)
    x = pd.concat([np.log(series).diff(i) for i in [5, 15, 30, 60]],axis=1).dropna()
    x.columns = [symbol+'_DT',symbol+'3DT',symbol+'6DT',symbol+'_12DT']
    dataset = pd.concat([x,y],axis=1).dropna().iloc[::5,:]
    y = dataset.loc[:,y.name]
    x = dataset.loc[:,x.columns]
    
    n_test = forecast_horizon
    x_train, x_test = x[0:-n_test], x[-n_test:len(x)]
    y_train, y_test = y[:-n_test], y[-n_test:]

    arima_train = y[:-n_test]
    arima_test = y[-n_test:]

    # Train ARIMA
    out['models']['arima'], out['tests']['arima_ljungbox'], out['tests']['arima_arch'] = build_arima(arima_train, n_test, arima_test)
    #plot_forecast(np.exp(arima_train).cumprod(), np.exp(arima_test).cumprod(), out['models']['arima']['returns'], out['models']['arima']['lower'], out['models']['arima']['upper'], title=f"{symbol} - ARIMA Forecast")
    out['models']['lstm'], out['tests']['lstm_ljungbox'], out['tests']['lstm_arch'] = build_lstm(x_train, y_train, x_test, y_test)
    #plot_forecast(np.exp(arima_train).cumprod(), np.exp(arima_test).cumprod(), out['models']['lstm']['returns'], out['models']['lstm']['lower'], out['models']['lstm']['upper'], title=f"{symbol} - LSTM Forecast")

    lstm_score = out['models']['lstm']['score']
    arima_score = out['models']['lstm']['score']
    total = np.exp(lstm_score)+np.exp(arima_score)
    lstm_weight = np.exp(lstm_score)/total
    arima_weight = np.exp(arima_score)/total

    lstm_preds = np.array(out['models']['lstm']['preds'])
    arima_preds = np.array(out['models']['arima']['preds'])
    lstm_returns = np.array(out['models']['lstm']['returns'])
    arima_returns = np.array(out['models']['arima']['returns'])
    lstm_lower = np.array(out['models']['lstm']['lower'])
    arima_lower = np.array(out['models']['arima']['lower'])
    lstm_upper = np.array(out['models']['lstm']['upper'])
    arima_upper = np.array(out['models']['arima']['upper'])

    arima_model = out['models']['arima']['model_name']
    lstm_model = out['models']['lstm']['model_name']
    model_name = arima_model + '-' + lstm_model
    out['models'][model_name] = {'model_name':'ensemble',out['models']['arima']['model_name']:out['models']['arima']['meta'],
                                out['models']['lstm']['model_name']:out['models']['lstm']['meta']}
    out['models'][model_name]['preds'] = lstm_weight * lstm_preds + arima_preds * arima_preds
    out['models'][model_name]['returns'] = lstm_weight * lstm_returns + arima_weight * arima_returns
    out['models'][model_name]['lower'] = lstm_weight * lstm_lower + arima_weight * arima_lower
    out['models'][model_name]['upper'] = lstm_weight * lstm_upper + arima_weight * arima_upper
    # evaluate against true prices
    out['models'][model_name]['metrics'] = compute_metrics(y_test.values, out['models'][model_name]['preds'])
    out['models'][model_name]['ci_cov'] = compute_ci_coverage(np.exp(y_test.values).cumprod(), out['models'][model_name]['lower'], out['models'][model_name]['upper'])
    out['models'][model_name]['score'] = reliability_score(out['models'][model_name]['metrics'], out['models'][model_name]['ci_cov'])

    # Select best model by score where available
    best_name, best_info = None, None
    best_score = -np.inf
    for name, info in out['models'].items():
        if isinstance(info, dict) and 'score' in info:
            if info['score'] > best_score:
                best_score = info['score']
                best_name = name
                best_info = info
    if best_name == 'ensemble':
        best_model = out['models']['ensemble']
    elif best_name == 'arima':
        best_model = out['models']['arima']
    else:
        best_model = out['models']['lstm']
    #plot_forecast(np.exp(arima_train).cumprod(), np.exp(arima_test).cumprod(), out['models']['ensemble']['returns'], out['models']['ensemble']['lower'], out['models']['ensemble']['upper'], title=f"{symbol} - ensemble Forecast")
    
    return best_model

# load_dataset
df = pd.read_csv('datasets/stock_data.csv')
df = df.set_index('Date')
results = {}
for symbol in df.columns:
    symbol1 = symbol.split('_')[1]
    results[symbol1] = pipeline_for_stock(symbol, df)

#create dataframe
rows = []
for symbol, info in results.items():
    model_name = info.get('model_name')
    metrics = info.get('metrics')
    model_obj = info.get('meta')
    
    row = {
        'symbol': symbol,
        'model_name': model_name,
        'model_obj': model_obj,
        'rmse': metrics.get('rmse'),
        'mape': metrics.get('mape'),
        'resid_std': metrics.get('resid_std'),
        'score': info.get('score'),
        'returns': info.get('returns'),
        'preds': info.get('preds'),
        'lower': info.get('lower'),
        'upper': info.get('upper'),
    }
    rows.append(row)

models_data = pd.DataFrame(rows)

with open("datasets/models_data.pkl", "wb") as f:
    pickle.dump(models_data, f)
