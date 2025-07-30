from flask import Flask, request, jsonify
import pandas as pd
import yfinance as yf
from forecasting_module import create_forecast
from datetime import timedelta, date
import numpy as np

app = Flask(__name__)

AVAILABLE_BANKS = {
    "BRIS": {"name": "Bank Syariah Indonesia", "ticker": "BRIS.JK"},
    "BTPS": {"name": "Bank BTPN Syariah", "ticker": "BTPS.JK"},
    "PNBS": {"name": "Bank Panin Dubai Syariah", "ticker": "PNBS.JK"},
    "BANK": {"name": "Bank Aladin Syariah", "ticker": "BANK.JK"}
}

@app.route('/forecast', methods=['GET'])
def forecast():
    bank_code = request.args.get('bank')
    days = request.args.get('days', default=30, type=int)
    start_date_str = request.args.get('start_date')

    if not bank_code or bank_code.upper() not in AVAILABLE_BANKS:
        return jsonify({"error": "Invalid or missing 'bank' parameter."}), 400

    if not start_date_str:
        return jsonify({"error": "Parameter 'start_date' is required."}), 400

    try:
        start_date = pd.to_datetime(start_date_str)
    except Exception:
        return jsonify({"error": "Invalid date format for 'start_date'. Use YYYY-MM-DD format."}), 400

    ticker = AVAILABLE_BANKS[bank_code.upper()]["ticker"]
    stock = yf.Ticker(ticker)

    # === Create Forecast ===
    predictions = create_forecast(bank_code.upper(), days=days, start_date=start_date_str)

    if isinstance(predictions, dict) and "error" in predictions:
        return jsonify(predictions), 500

    df_pred = pd.DataFrame(predictions)
    df_pred['date'] = pd.to_datetime(df_pred['date']).dt.strftime('%Y-%m-%d')

    # === Ambil Data Historis ===
    df_actual = pd.DataFrame()
    try:
        end_date = start_date + timedelta(days=days * 2)
        df_actual_raw = stock.history(start=start_date_str, end=end_date.strftime('%Y-%m-%d'), auto_adjust=True)

        if not df_actual_raw.empty:
            df_actual_raw.reset_index(inplace=True)
            date_column = 'Datetime' if 'Datetime' in df_actual_raw.columns else 'Date'
            df_actual_raw.rename(columns={'Close': 'actual_close'}, inplace=True)
            df_actual = df_actual_raw.copy()
            df_actual['date'] = df_actual[date_column].dt.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Could not fetch historical data for {ticker}. Error: {e}")

    # === Gabungkan prediksi dan data aktual ===
    if not df_actual.empty:
        df_comparison = pd.merge(
            df_pred,
            df_actual[['date', 'actual_close']],
            on='date',
            how='left'
        )
    else:
        df_comparison = df_pred
        df_comparison['actual_close'] = None

    # === Ambil harga live (jika tersedia) ===
    try:
        today_str = date.today().strftime('%Y-%m-%d')
        if today_str in df_comparison['date'].values:
            df_today_live = stock.history(period='1d', interval='15m', auto_adjust=True)
            if not df_today_live.empty:
                live_price = df_today_live['Close'].iloc[-1]
                df_comparison.loc[df_comparison['date'] == today_str, 'actual_close'] = live_price
    except Exception as e:
        print(f"Could not fetch live price for {ticker}. It's likely the market is closed. Error: {e}")

    df_comparison.replace({np.nan: None}, inplace=True)
    comparison_data = df_comparison.to_dict(orient='records')

    return jsonify({
        "bank": bank_code.upper(),
        "start_date": start_date_str,
        "comparison_data": comparison_data
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
        df = stock.history(period=range_param, interval=interval_param, auto_adjust=True)

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
        print(f"Error saat mengambil data historis untuk {ticker_symbol}: {e}")
        return jsonify({"error": "An error occurred while fetching historical data."}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
