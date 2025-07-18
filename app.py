from flask import Flask, request, jsonify
import traceback
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pmdarima as pm
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing

app = Flask(__name__)

AVAILABLE_BANKS = {
    "BRIS": {"name": "Bank Syariah Indonesia", "ticker": "BRIS.JK"},
    "BTPS": {"name": "Bank BTPN Syariah", "ticker": "BTPS.JK"},
    "PNBS": {"name": "Bank Panin Dubai Syariah", "ticker": "PNBS.JK"},
    "BANK": {"name": "Bank Aladin Syariah", "ticker": "BANK.JK"}
}

# Model Forecasting (ARIMA + LSTM + Smoothing + Hybrid)

def get_stock_data(ticker, start_date='2021-02-01', end_date='2024-07-31'):
    print(f"⏳ Mengambil data untuk: {ticker}")
    df = yf.download(ticker, start=start_date, end=end_date)
    return df[['Close']]

def normalize_data(data):
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
    print("🔁 Melatih model ARIMA...")
    model = pm.auto_arima(data, seasonal=False, suppress_warnings=True, stepwise=True, error_action='ignore')
    return model

def train_lstm_model(X, y):
    print("🔁 Melatih model LSTM...")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)
    return model

def train_smoothing_model(data, forecast_steps):
    print("🔁 Melatih model Smoothing Regression")
    smoothed = sm.nonparametric.lowess(data, np.arange(len(data)), frac=0.1)[:, 1]
    model = ExponentialSmoothing(smoothed, trend='add').fit()
    return model.forecast(steps=forecast_steps)

def forecast_stock(ticker, days=30, use_hybrid=True, start_date=None):
    print(f"📊 Memproses forecasting untuk {ticker} mulai dari {start_date} selama {days} hari...")

    df = get_stock_data(ticker)
    if df.empty or len(df) < 100:
        return {"error": f"Data untuk {ticker} tidak cukup atau tidak tersedia."}

    close_prices = df['Close'].values.reshape(-1, 1)
    scaled_data, scaler = normalize_data(close_prices)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    total_forecast_steps = len(test_data) + days

    # ARIMA
    arima_model = train_arima_model(train_data.flatten())
    pred_arima_scaled = arima_model.predict(n_periods=total_forecast_steps)
    pred_arima = scaler.inverse_transform(pred_arima_scaled.reshape(-1, 1))

    # LSTM
    seq_length = 10
    X_train, y_train = create_sequences(train_data, seq_length)
    lstm_model = train_lstm_model(X_train, y_train)

    lstm_preds_scaled = []
    current_seq = scaled_data[-seq_length:].reshape(seq_length, 1)
    for _ in range(total_forecast_steps):
        pred = lstm_model.predict(current_seq.reshape(1, seq_length, 1), verbose=0)
        lstm_preds_scaled.append(pred[0][0])
        current_seq = np.vstack([current_seq[1:], [[pred[0][0]]]])

    pred_lstm = scaler.inverse_transform(np.array(lstm_preds_scaled).reshape(-1, 1))

    if not use_hybrid:
        start_date = pd.to_datetime(start_date or "2025-01-01")
        forecast_dates = pd.date_range(start=start_date, periods=days)
        return [
            {"date": d.strftime("%Y-%m-%d"), "predicted_close": round(v[0], 2)}
            for d, v in zip(forecast_dates, pred_lstm[-days:])
        ]

    # Smoothing Regression
    pred_smooth_scaled = train_smoothing_model(train_data.flatten(), total_forecast_steps)
    pred_smooth = scaler.inverse_transform(pred_smooth_scaled.reshape(-1, 1))

    # Hybrid OLS
    print("🔁 Menggabungkan prediksi ARIMA, LSTM, dan Smoothing dengan Regresi Linear...")
    min_len = min(len(pred_arima), len(pred_lstm), len(pred_smooth))
    pred_arima = pred_arima[:min_len]
    pred_lstm = pred_lstm[:min_len]
    pred_smooth = pred_smooth[:min_len]

    hybrid_input = np.column_stack((pred_arima, pred_lstm, pred_smooth))
    hybrid_input = sm.add_constant(hybrid_input)

    y_actual = close_prices[train_size:train_size + min_len]
    hybrid_input_train = hybrid_input[:len(y_actual)]
    reg_model = sm.OLS(y_actual, hybrid_input_train).fit()
    final_preds = reg_model.predict(hybrid_input[-days:])

    start_date = pd.to_datetime(start_date or "2025-01-01")
    forecast_dates = pd.date_range(start=start_date, periods=days)
    result = [
        {"date": d.strftime("%Y-%m-%d"), "predicted_close": round(v, 2)}
        for d, v in zip(forecast_dates, final_preds)
    ]
    return result

@app.route('/forecast', methods=['GET'])
def forecast():
    bank_code = request.args.get('bank')
    days = request.args.get('days', default=30, type=int)
    start_date = request.args.get('start_date')

    if not bank_code or bank_code.upper() not in AVAILABLE_BANKS:
        return jsonify({"error": "Invalid or missing 'bank' parameter."}), 400

    if not start_date:
        return jsonify({"error": "Parameter 'start_date' is required."}), 400

    ticker = AVAILABLE_BANKS[bank_code.upper()]["ticker"]

    try:
        pd.to_datetime(start_date)
    except Exception:
        return jsonify({"error": "Invalid date format for 'start_date'. Use YYYY-MM-DD format."}), 400

    predictions = forecast_stock(ticker, days=days, start_date=start_date)

    if isinstance(predictions, dict) and "error" in predictions:
        return jsonify(predictions), 500

    return jsonify({
        "bank": bank_code.upper(),
        "days": days,
        "start_date": start_date,
        "predictions": predictions
    })

@app.route('/historical/<bank_code>', methods=['GET'])
def historical(bank_code):
    range_param = request.args.get('range', default='1mo')
    interval_param = request.args.get('interval', default='1d')

    if bank_code.upper() not in AVAILABLE_BANKS:
        return jsonify({"error": "Invalid bank code."}), 400

    ticker_symbol = AVAILABLE_BANKS[bank_code.upper()]["ticker"]

    try:
        if range_param == '1d' and interval_param == '1d':
            interval_param = '15m'

        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period=range_param, interval=interval_param)

        if df.empty:
            return jsonify([]), 200

        df.reset_index(inplace=True)
        date_column_name = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df['date_str'] = df[date_column_name].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['x'] = range(len(df))
        df.rename(columns={'Close': 'close', 'date_str': 'date'}, inplace=True)

        selected_data = df[['x', 'date', 'close']].to_dict(orient='records')
        return jsonify(selected_data)

    except Exception as e:
        print(f"❌ Error saat mengambil data historis untuk {ticker_symbol}: {e}")
        return jsonify({"error": "An error occurred while fetching historical data."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
