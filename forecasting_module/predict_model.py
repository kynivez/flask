# predict_model.py

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from keras.models import load_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances
import os

def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
    return np.array(X)

def lowess_2d(X1, X2, Y, x1_new, x2_new, frac=0.1):
    X = np.vstack((X1, X2)).T
    distances = euclidean_distances(X, np.array([[x1_new, x2_new]]))
    k = int(frac * len(X1))
    bandwidth = np.sort(distances, axis=0)[k]
    bandwidth = np.maximum(bandwidth, 1e-8)
    weights = (1 - (distances / bandwidth) ** 3) ** 3
    weights[distances > bandwidth] = 0
    reg = LinearRegression()
    reg.fit(X, Y, sample_weight=weights.flatten())
    return reg.predict(np.array([[x1_new, x2_new]]))[0]

def lowess_predict_all(X1, X2, Y, frac=0.1):
    return np.array([lowess_2d(X1, X2, Y, x1, x2, frac) for x1, x2 in zip(X1, X2)])

def create_forecast(bank_code, days=30, start_date='2024-07-01'):
    bank_models = {
        "BRIS": "BRIS",
        "BTPS": "BTPS",
        "PNBS": "PNBS",
        "BANK": "BANK"
    }

    if bank_code not in bank_models:
        return {"error": f"Model for {bank_code} not found."}

    model_dir = f"saved_models/{bank_models[bank_code]}"
    try:
        arima = joblib.load(os.path.join(model_dir, "arima_model.pkl"))
        lstm = load_model(os.path.join(model_dir, "lstm_model.keras"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}

    ticker = f"{bank_code}.JK"
    start_date = pd.to_datetime(start_date)
    history_days = 60
    end_date = start_date - pd.Timedelta(days=1)
    start_history = end_date - pd.Timedelta(days=history_days)
    df_hist = yf.download(ticker, start=start_history.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    if df_hist.empty:
        return {"error": "No historical data available for forecasting."}

    close = df_hist['Close'].values.reshape(-1, 1)
    scaled = scaler.transform(close)
    seq_length = 1

    # ARIMA
    arima_forecast = scaler.inverse_transform(arima.predict(n_periods=days).reshape(-1, 1)).flatten()

    # LSTM
    lstm_input_seq = create_sequences(scaled, seq_length)[-1:]
    lstm_forecast_scaled = []
    for _ in range(days):
        pred = lstm.predict(lstm_input_seq)[0][0]
        lstm_forecast_scaled.append(pred)
        lstm_input_seq = np.append(lstm_input_seq[:, 1:, :], [[[pred]]], axis=1)
    lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast_scaled).reshape(-1, 1)).flatten()

    # Actual past values
    last_actual = close[-days:].flatten()
    if len(last_actual) < days:
        last_actual = np.pad(last_actual, (days - len(last_actual), 0), mode='edge')

    # Lowess hybrid
    hybrid_forecast = lowess_predict_all(arima_forecast, lstm_forecast, last_actual, frac=0.2)

    # Build output
    dates = pd.date_range(start=start_date, periods=days).strftime('%Y-%m-%d')
    forecast = [{"date": date, "predicted_close": float(pred)} for date, pred in zip(dates, hybrid_forecast)]
    return forecast
