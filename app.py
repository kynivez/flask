from flask import Flask, request, jsonify
import pandas as pd
import yfinance as yf
from forecasting_module import create_forecast

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

    predictions = create_forecast(bank_code.upper(), days=days, start_date=start_date)

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
