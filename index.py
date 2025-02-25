from flask import Flask, request, jsonify
import tushare as ts
import pandas as pd
import numpy as np
import talib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS
from datetime import datetime, timedelta

from tokens import TUSHARE_TOKEN

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def get_stock_data(stock_code):
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    start_date = '20240101'
    df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    df = df.sort_index()
    df = df[['close', 'open', 'high', 'low', 'vol']]
    df['MA5'] = talib.SMA(df['close'], timeperiod=5)
    df['MA10'] = talib.SMA(df['close'], timeperiod=10)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], _, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['OBV'] = talib.OBV(df['close'], df['vol'])
    df['upperband'], df['middleband'], df['lowerband'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['vol'], timeperiod=14)
    df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df = df.bfill()
    return df

def train_and_predict(df):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    look_back = 10
    features = df_scaled.shape[1]

    def create_lstm_dataset(data, look_back):
        X, Y = [], []
        for i in range(len(data) - look_back):
            X.append(data.iloc[i:i + look_back].values)
            Y.append(data.iloc[i + look_back]['close'])
        return np.array(X), np.array(Y)

    X, y = create_lstm_dataset(df_scaled, look_back)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential([
        Input(shape=(look_back, features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0)

    future_predictions_runs = []
    for _ in range(5):
        last_data = df_scaled[-look_back:].values.reshape(1, look_back, features)
        future_predictions = []
        for _ in range(10):
            pred = model.predict(last_data)
            future_predictions.append(pred[0, 0])
            last_data_2d = last_data.reshape(look_back, features)
            new_row = last_data_2d[-1].copy()
            close_index = df.columns.get_loc('close')
            new_row[close_index] = pred[0, 0]
            new_window = np.vstack([last_data_2d[1:], new_row.reshape(1, features)])
            last_data = new_window.reshape(1, look_back, features)
        future_predictions_runs.append(future_predictions)

    future_predictions_avg = np.mean(future_predictions_runs, axis=0)
    close_index = df.columns.get_loc('close')
    close_min = scaler.min_[close_index]
    close_scale = scaler.scale_[close_index]
    future_predictions_actual = (future_predictions_avg - close_min) / close_scale

    future_dates = pd.bdate_range(start=df_scaled.index[-1] + pd.Timedelta(days=1), periods=10).strftime('%Y-%m-%d')
    result = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions_actual.flatten()})
    result.set_index('Date', inplace=True)

    trend = "上涨" if result.iloc[0]['Predicted_Close'] > df.iloc[-1]['close'] else "下跌"
    return result, trend

@app.route('/predict', methods=['GET'])
def predict():
    stock_code = request.args.get('stock_code')
    if not stock_code:
        return jsonify({'error': 'Stock code is required'}), 400
    df = get_stock_data(stock_code)
    result, trend = train_and_predict(df)
    response = result.reset_index().to_dict(orient='records')
    return jsonify({"trend": trend, "predictions": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5800, debug=True)