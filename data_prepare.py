import pandas as pd
import requests
import time
import numpy as np
from datetime import datetime, timedelta
import io

API_KEY = 'PUDPSYYSPAF8IGTR'
SYMBOL = 'SPY'
INTERVAL = '1min'

def get_intraday_data(symbol, interval, api_key, start_date, end_date):
    data_frames = []
    current_date = start_date

    while current_date <= end_date:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}&outputsize=full'
        print(f'Fetching data starting from {current_date.strftime("%Y-%m-%d")}')
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'Time Series (1min)' in data:
                df = pd.DataFrame.from_dict(data['Time Series (1min)'], orient='index')
                df = df.rename(columns={
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. volume': 'Volume'
                })
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)

                # 筛选出指定时间范围内的数据
                df = df[(df.index >= current_date) & (df.index <= end_date)]
                
                if not df.empty:
                    latest_timestamp = df.index.max()
                    current_date = latest_timestamp + timedelta(minutes=1)
                    data_frames.append(df)
                else:
                    break
            else:
                print(f"Error: {data}")
                break
        else:
            print(f"HTTP Error: {response.status_code}")
            break
        
        time.sleep(12)  # 避免API速率限制

    if data_frames:
        full_data = pd.concat(data_frames)
        return full_data
    else:
        return pd.DataFrame()  # 返回空的 DataFrame 以防止后续代码出错

def get_full_data(symbol, interval, api_key, start_date=None, end_date=None, days=None):
    if days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
    elif not (start_date and end_date):
        raise ValueError("Either days or both start_date and end_date must be specified")
    return get_intraday_data(symbol, interval, api_key, start_date, end_date)

# 获取数据
days_to_fetch = 30  # 用户可修改的参数，表示获取多少天的数据
start_date = None  # 可以设置为用户指定的起始日期，例如 datetime(2023, 1, 1)
end_date = None  # 可以设置为用户指定的结束日期，例如 datetime(2023, 6, 30)
df = get_full_data(SYMBOL, INTERVAL, API_KEY, start_date=start_date, end_date=end_date, days=days_to_fetch)

if not df.empty:
    # 重命名列并排序
    df = df.sort_index()

    # 数据清洗：删除重复数据
    df = df.drop_duplicates()

    # 特征工程
    def get_technical_indicators(df):
        df['SMA'] = df['Close'].rolling(window=15).mean()
        df['EMA'] = df['Close'].ewm(span=15, adjust=False).mean()
        df['RSI'] = compute_rsi(df['Close'], 14)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(df['Close'])
        return df

    def compute_rsi(series, period):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        RS = gain / loss
        return 100 - (100 / (1 + RS))

    def compute_macd(series, slow=26, fast=12, signal=9):
        fast_ema = series.ewm(span=fast, min_periods=1, adjust=False).mean()
        slow_ema = series.ewm(span=slow, min_periods=1, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal, min_periods=1, adjust=False).mean()
        macd_hist = macd - signal_line
        return macd, signal_line, macd_hist

    df = get_technical_indicators(df)

    # 增加更多特征
    df['Price_Change'] = df['Close'].pct_change()
    df['High_Low_Spread'] = df['High'] - df['Low']
    df['Close_Open_Spread'] = df['Close'] - df['Open']
    df['Open_Prev_Close_Spread'] = df['Open'] - df['Close'].shift(1)

    # 数据清洗：处理缺失值
    df.dropna(inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # 保存处理后的数据
    df.to_csv('data/SPY_intraday_cleaned.csv')

    # 打印数据样本以确认
    print(df.head())
else:
    print("No data fetched. Please check the parameters or API limits.")
