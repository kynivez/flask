import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pmdarima as pm
import statsmodels.api as sm
import joblib
import os

# Mapping kode bank ke ticker saham
BANKS = {
    "BRIS": "BRIS.JK",
    "BTPS": "BTPS.JK",
    "PNBS": "PNBS.JK",
    "BANK": "BANK.JK"
}

# Mengambil data penutupan saham dari Yahoo Finance
def get_data(ticker):
    df = yf.download(ticker, start='2021-02-01', end='2024-07-31')
    return df[['Close']]

# Normalisasi menggunakan MinMaxScaler
def normalize(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled, scaler

# Membentuk urutan data untuk LSTM
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Melatih model ARIMA
def train_arima_model(data):
    # Menggunakan auto_arima untuk menemukan parameter ARIMA terbaik
    return pm.auto_arima(data, seasonal=False, suppress_warnings=True, stepwise=True, error_action='ignore')

# Melatih model LSTM
def train_lstm_model(X, y):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1)) # Outputnya sekarang adalah prediksi residual
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)
    return model

# Lowess smoothing regression (tidak berubah, untuk visualisasi)
def train_lowess_model(data):
    smoothed = sm.nonparametric.lowess(data.flatten(), np.arange(len(data)), frac=0.1)[:, 1]
    return smoothed

# Menyimpan semua model dan data yang dibutuhkan
def save_all_models(bank_code, arima_model, lstm_model, scaler, close_prices, lowess_data):
    model_dir = f"models/{bank_code}"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(arima_model, f"{model_dir}/arima_model.pkl")
    lstm_model.save(f"{model_dir}/lstm_model.h5")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    np.save(f"{model_dir}/close_prices.npy", close_prices)
    np.save(f"{model_dir}/lowess.npy", lowess_data)

# Training untuk seluruh bank
if __name__ == "__main__":
    for code, ticker in BANKS.items():
        print(f"Melatih model hybrid untuk {code}")
        df = get_data(ticker)
        close_prices = df['Close'].values.reshape(-1, 1)
        
        # Normalisasi dan split
        scaled, scaler = normalize(close_prices)
        train_size = int(len(scaled) * 0.8)
        train_data = scaled[:train_size]

        # --- LANGKAH 1: Latih ARIMA ---
        arima_model = train_arima_model(train_data.flatten())

        # --- LANGKAH 2: Hitung Residual dari ARIMA ---
        # Dapatkan prediksi ARIMA pada data latihan
        predictions_in_sample = arima_model.predict_in_sample()
        # Hitung selisih (residual)
        residuals = train_data.flatten() - predictions_in_sample
        
        # --- LANGKAH 3: Siapkan Data untuk LSTM ---
        # Input (X) tetap dari data harga, tapi target (y) adalah residual
        X_train, _ = create_sequences(train_data)
        # y_train sekarang adalah urutan residual yang sesuai
        y_train_residuals = residuals[10:] # Mulai dari indeks ke-10 (sesuai seq_length)

        # Pastikan panjang X_train dan y_train_residuals sama
        if len(X_train) > len(y_train_residuals):
            X_train = X_train[:len(y_train_residuals)]
        
        # --- LANGKAH 4: Latih LSTM pada Residual ---
        lstm_model = train_lstm_model(X_train, y_train_residuals)

        # Training Lowess (tidak berubah)
        lowess_data = train_lowess_model(close_prices)

        # Simpan semua komponen
        save_all_models(code, arima_model, lstm_model, scaler, close_prices, lowess_data)
        print(f"Model hybrid untuk {code} disimpan.")
