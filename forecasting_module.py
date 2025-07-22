# forecasting_module.py

import numpy as np
import pandas as pd
import joblib
import os
import statsmodels.api as sm
from keras.models import load_model
from statsmodels.tsa.api import ExponentialSmoothing

def load_models(bank_code):
    base_path = f"models/{bank_code.upper()}"
    arima = joblib.load(f"{base_path}/arima_model.pkl")
    lstm = load_model(f"{base_path}/lstm_model.h5")
    scaler = joblib.load(f"{base_path}/scaler.pkl")
    close_prices = np.load(f"{base_path}/close_prices.npy")
    return arima, lstm, scaler, close_prices

def create_forecast(bank_code, days, start_date, use_hybrid=True):
    arima, lstm, scaler, close_prices = load_models(bank_code)
    scaled = scaler.transform(close_prices)

    seq_length = 10
    current_seq = scaled[-seq_length:].reshape(seq_length, 1)

    lstm_preds = []
    for _ in range(days):
        pred = lstm.predict(current_seq.reshape(1, seq_length, 1), verbose=0)
        lstm_preds.append(pred[0][0])
        current_seq = np.vstack([current_seq[1:], [[pred[0][0]]]])

    lstm_pred = scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1))

    if not use_hybrid:
        forecast_dates = pd.date_range(start=start_date, periods=days)
        return [
            {"date": d.strftime("%Y-%m-%d"), "predicted_close": round(v[0], 2)}
            for d, v in zip(forecast_dates, lstm_pred)
        ]

    # Smoothing Regression
    train_data = scaled[:int(len(scaled) * 0.8)]
    smoothed = sm.nonparametric.lowess(train_data.flatten(), np.arange(len(train_data)), frac=0.1)[:, 1]
    smooth_model = ExponentialSmoothing(smoothed, trend='add').fit()
    smooth_pred = scaler.inverse_transform(smooth_model.forecast(days).reshape(-1, 1))

    arima_pred = scaler.inverse_transform(arima.predict(n_periods=days).reshape(-1, 1))

    # Gabung hybrid
    hybrid_input = np.column_stack((arima_pred, lstm_pred, smooth_pred))
    hybrid_input = sm.add_constant(hybrid_input)
    y_actual = close_prices[-days:]
    reg_model = sm.OLS(y_actual, hybrid_input).fit()
    final_pred = reg_model.predict(hybrid_input)

    forecast_dates = pd.date_range(start=start_date, periods=days)
    return [
        {"date": d.strftime("%Y-%m-%d"), "predicted_close": round(v, 2)}
        for d, v in zip(forecast_dates, final_pred)
    ]
