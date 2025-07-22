# train_model.py

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pmdarima as pm
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing
import joblib
import os

BANKS = {
    "BRIS": "BRIS.JK",
    "BTPS": "BTPS.JK",
    "PNBS": "PNBS.JK",
    "BANK": "BANK.JK"
}

def get_data(ticker):
    df = yf.download(ticker, start='2021-02-01', end='2024-07-31')
    return df[['Close']]

def normalize(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled, scaler

def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_arima_model(data):
    return pm.auto_arima(data, seasonal=False, suppress_warnings=True, stepwise=True, error_action='ignore')

def train_lstm_model(X, y):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)
    return model

def train_smoothing_model(data, steps):
    smoothed = sm.nonparametric.lowess(data, np.arange(len(data)), frac=0.1)[:, 1]
    model = ExponentialSmoothing(smoothed, trend='add').fit()
    return model.forecast(steps)

def save_all_models(bank_code, arima_model, lstm_model, scaler, close_prices):
    model_dir = f"models/{bank_code}"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(arima_model, f"{model_dir}/arima_model.pkl")
    lstm_model.save(f"{model_dir}/lstm_model.h5")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    np.save(f"{model_dir}/close_prices.npy", close_prices)

if __name__ == "__main__":
    for code, ticker in BANKS.items():
        print(f"🚀 Melatih model untuk {code}")
        df = get_data(ticker)
        close_prices = df['Close'].values.reshape(-1, 1)
        scaled, scaler = normalize(close_prices)
        train_size = int(len(scaled) * 0.8)
        train_data = scaled[:train_size]

        arima_model = train_arima_model(train_data.flatten())

        X_train, y_train = create_sequences(train_data)
        lstm_model = train_lstm_model(X_train, y_train)

        save_all_models(code, arima_model, lstm_model, scaler, close_prices)
        print(f"✅ Model untuk {code} disimpan.")
