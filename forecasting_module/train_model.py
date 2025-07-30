# train_model.py

import os
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pmdarima as pm
from keras.models import Sequential
from keras.layers import LSTM, Dense
import joblib

# ==== Konfigurasi Bank dan Ticker ====
BANKS = {
    "BRIS": "BRIS.JK",
    "BTPS": "BTPS.JK",
    "PNBS": "PNBS.JK",
    "BANK": "BANK.JK"
}

# ==== Fungsi Bantu ====
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_bank_model(bank_code, ticker, start='2021-02-01', end='2024-07-01'):
    print(f"\n=== Melatih model untuk {bank_code} ({ticker}) ===")
    output_dir = f"saved_models/{bank_code}"
    os.makedirs(output_dir, exist_ok=True)

    # Ambil data historis
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        print(f"❌ Data kosong untuk {bank_code}")
        return

    close_data = data['Close'].values.reshape(-1, 1)

    # Normalisasi
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_data)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]

    # === ARIMA ===
    try:
        arima_model = pm.auto_arima(train_data, seasonal=False, trace=False, suppress_warnings=True)
        joblib.dump(arima_model, os.path.join(output_dir, 'arima_model.pkl'))
        print("✅ ARIMA model disimpan.")
    except Exception as e:
        print(f"❌ Gagal melatih ARIMA untuk {bank_code}: {e}")
        return

    # === LSTM ===
    seq_length = 1
    X_train, y_train = create_sequences(train_data, seq_length)

    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    lstm_model.save(os.path.join(output_dir, 'lstm_model.keras'))
    print("✅ LSTM model disimpan.")

    # === Scaler ===
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print("✅ Scaler disimpan.")

# ==== Jalankan pelatihan untuk semua bank ====
if __name__ == "__main__":
    for bank_code, ticker in BANKS.items():
        train_bank_model(bank_code, ticker)
