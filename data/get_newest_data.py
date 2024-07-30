import pandas as pd
import os
import requests
from datetime import datetime, timedelta
import time
import io

API_KEY = 'PUDPSYYSPAF8IGTR'
SYMBOL = 'SPY'
INTERVAL = '1min'
API_CALLS_LIMIT = 5

def get_intraday_data(symbol, interval, api_key, start_date):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}&outputsize=full'
    print(f'Fetching data from {start_date} to now')
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
            df = df[df.index >= start_date]
            return df.astype(float)
        else:
            print(f"Error: {data}")
            return pd.DataFrame()
    else:
        print(f"HTTP Error: {response.status_code}")
        return pd.DataFrame()

def get_advanced_technical_indicators(df):
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['RSI_Change'] = df['RSI'].diff()
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(df['Close'])
    df['MACD_Change'] = df['MACD'].diff()
    df['BB_High'], df['BB_Low'] = compute_bollinger_bands(df['Close'])
    df['BB_Width'] = df['BB_High'] - df['BB_Low']
    df['Historical_Volatility'] = df['Close'].rolling(window=21).std() * (252 ** 0.5)
    df['Price_Volume'] = df['Close'] * df['Volume']
    df['Momentum'] = df['Close'].diff(4)
    df['Stochastic_Oscillator'] = ((df['Close'] - df['Low'].rolling(window=14).min()) /
                                   (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())) * 100
    df['Gap'] = df['Open'] - df['Previous_Close']
    df.dropna(inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
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

def compute_bollinger_bands(series, window=20, num_std_dev=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def update_daily_data(symbol, interval, api_key):
    # 获取最新的数据目录
    data_dir = 'data/daily_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 找到最新的日期文件
    files = os.listdir(data_dir)
    if not files:
        print("No existing data found. Please run the initial data fetching process first.")
        return
    
    latest_file = max(files)
    latest_date_str = latest_file.split('_')[-1].split('.')[0]
    latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
    next_date = latest_date + timedelta(days=1)

    # 获取从最新日期到今天的数据
    end_date = datetime.now()
    df_new = get_intraday_data(symbol, interval, api_key, next_date)
    if df_new.empty:
        print("No new data fetched.")
        return

    # 添加Previous_Close列
    df_new['Previous_Close'] = df_new['Close'].shift(1)
    
    # 进行复杂特征工程
    df_new = get_advanced_technical_indicators(df_new)

    # 按天切分并保存
    for date, group in df_new.groupby(df_new.index.date):
        output_file = os.path.join(data_dir, f'{symbol}_intraday_{date}.csv')
        group.to_csv(output_file)

    print(f"New data has been fetched and saved to {data_dir}")

# 更新数据
update_daily_data(SYMBOL, INTERVAL, API_KEY)
