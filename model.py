import os
import pandas as pd
import numpy as np
import tushare as ts
import talib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pickle

from tokens import TUSHARE_TOKEN  # 导入你的 Tushare API Token

# **设置 Tushare API**
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# **存储路径**
MODEL_PATH = "universal_lstm_model.h5"  # 训练好的 LSTM 模型存放路径
SCALER_PATH = "scaler.pkl"  # MinMaxScaler 归一化参数存放路径


def get_stock_data(stock_code, start_date="20240101"):
    """
    获取指定股票的历史数据，并计算技术指标。

    :param stock_code: 股票代码 (如 "000001.SZ")
    :param start_date: 数据起始日期，默认从 2024 年 1 月 1 日开始
    :return: 处理后的 DataFrame（包含价格数据和技术指标），如果数据为空，则返回 None
    """
    # **获取数据的结束时间（默认是昨天）**
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

    # **从 Tushare 获取历史数据**
    df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)

    if df is None or df.empty:
        return None  # 如果没有数据，返回 None

    # **转换 trade_date 为时间格式，并设置为索引**
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    df = df.sort_index()  # 按时间升序排列

    # **选取关键信息**
    df = df[['close', 'open', 'high', 'low', 'vol']]

    # **计算技术指标**
    df['MA5'] = talib.SMA(df['close'], timeperiod=5)  # 5 日均线
    df['MA10'] = talib.SMA(df['close'], timeperiod=10)  # 10 日均线
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)  # 相对强弱指数（RSI）
    df['MACD'], _, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)  # MACD 指标
    df['OBV'] = talib.OBV(df['close'], df['vol'])  # 资金流量指标
    df['upperband'], df['middleband'], df['lowerband'] = talib.BBANDS(df['close'], timeperiod=20)  # 布林带
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)  # ADX 指标
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)  # CCI 指标
    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['vol'], timeperiod=14)  # 资金流量指数
    df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'])  # 随机震荡指标

    # **填充空值（使用前值填充）**
    df = df.bfill()

    return df


def train_universal_model():
    """
    训练通用 LSTM 股票预测模型（适用于所有股票）。
    训练数据包括多个热门股票，生成一个可以泛化的 LSTM 预测模型。

    :return: None（训练完成后保存模型）
    """
    stock_list = ["000001.SZ", "600519.SH", "002594.SZ"]  # 选取一些热门股票作为训练数据
    all_data = []

    for stock in stock_list:
        df = get_stock_data(stock)
        if df is not None:
            all_data.append(df)

    if not all_data:
        print("❌ 没有足够的股票数据训练模型！")
        return

    # **合并多个股票数据**
    df_all = pd.concat(all_data)

    # **数据归一化**
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_all), columns=df_all.columns, index=df_all.index)

    look_back = 10  # LSTM 需要过去 10 天的数据作为输入
    features = df_scaled.shape[1]  # 特征数量

    def create_lstm_dataset(data, look_back):
        """创建 LSTM 训练数据集"""
        X, Y = [], []
        for i in range(len(data) - look_back):
            X.append(data.iloc[i:i + look_back].values)
            Y.append(data.iloc[i + look_back]['close'])  # 目标值是未来的收盘价
        return np.array(X), np.array(Y)

    X, y = create_lstm_dataset(df_scaled, look_back)
    train_size = int(len(X) * 0.8)  # 80% 训练集，20% 测试集
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # **构建 LSTM 模型**
    model = Sequential([
        Input(shape=(look_back, features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')  # 均方误差（MSE）作为损失函数
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0)

    # **保存模型和归一化参数**
    model.save(MODEL_PATH)
    pickle.dump(scaler, open(SCALER_PATH, 'wb'))

    print("✅ 通用模型训练完成！")


def predict_stock(stock_code):
    """
    使用通用模型预测股票未来 10 天的收盘价。

    :param stock_code: 股票代码
    :return: 预测结果 DataFrame 和趋势（上涨/下跌）
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, "❌ 需要先训练通用模型！"

    # **加载模型**
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='mse')

    # **加载归一化器**
    scaler = pickle.load(open(SCALER_PATH, 'rb'))

    df = get_stock_data(stock_code)
    if df is None:
        return None, "❌ 股票代码无效或无数据！"

    # **归一化数据**
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

    look_back = 10
    features = df_scaled.shape[1]

    # **滚动预测未来 10 天**
    last_data = df_scaled[-look_back:].values.reshape(1, look_back, features)
    future_predictions = []

    for _ in range(10):
        pred = model.predict(last_data)
        future_predictions.append(pred[0, 0])

        last_data = np.vstack([last_data[0][1:], np.append(last_data[0][-1], pred[0, 0])]).reshape(1, look_back,
                                                                                                   features)

    future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=10).strftime('%Y-%m-%d')
    result = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions})
    result.set_index('Date', inplace=True)

    trend = "上涨" if result.iloc[0]['Predicted_Close'] > df.iloc[-1]['close'] else "下跌"
    return result, trend