# API_KEY = 'PUDPSYYSPAF8IGTR'
# API_KEY = 'QET9BN7YKRRNED2B'

import pandas as pd
import requests
import os
from datetime import datetime, timedelta
import time
import numpy as np

API_KEY = 'QET9BN7YKRRNED2B'
SYMBOL = 'SPY'
INTERVAL = '1min'
API_CALLS_LIMIT = 5  # AlphaVantage每分钟最多允许5次API调用

def get_intraday_data(symbol, interval, api_key, start_date, end_date):
    data_frames = []
    current_date = end_date

    url = "https://alpha-vantage.p.rapidapi.com/query"
    headers = {
        "x-rapidapi-key": "f3f2b6a315msh93ef2aea50a6658p1896e0jsn4fc95d596688",
        "x-rapidapi-host": "alpha-vantage.p.rapidapi.com"
    }

    while current_date >= start_date:
        year_month = current_date.strftime("%Y-%m")
        # url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}&outputsize=full&month={year_month}'

        querystring = {
            "function":"TIME_SERIES_INTRADAY",
            "symbol":SYMBOL,
            "interval":INTERVAL,
            "outputsize":"full",
            "month":year_month,
            "datatype":"json"
        }
        print(f'Fetching data for {year_month}')
        response = requests.get(url, headers=headers, params=querystring)
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
                data_frames.append(df)
            else:
                print(f"Unexpected data format for {year_month}: {data}")
        else:
            print(f"HTTP Error: {response.status_code}")

        current_date -= timedelta(days=30)
        time.sleep(60 / API_CALLS_LIMIT)  # 避免API速率限制

    if data_frames:
        full_data = pd.concat(data_frames)
        full_data = full_data[~full_data.index.duplicated(keep='first')]  # 删除重复数据
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

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    rsi = 100 - (100 / (1 + RS))
    return rsi

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

def compute_mfi(df, window=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    positive_mf = positive_flow.rolling(window=window).sum()
    negative_mf = negative_flow.rolling(window=window).sum()
    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    return mfi

def compute_cci(df, window=20):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    rolling_mean = typical_price.rolling(window=window).mean()
    rolling_std = typical_price.rolling(window=window).std()
    cci = (typical_price - rolling_mean) / (0.015 * rolling_std)
    return cci

def compute_true_range(df):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range

def compute_historical_volatility(df, window=14):
    log_return = np.log(df['Close'] / df['Close'].shift(1))
    volatility = log_return.rolling(window=window).std() * np.sqrt(252)
    return volatility

def compute_price_volume(df):
    price_volume = df['Close'] * df['Volume']
    return price_volume

def compute_momentum(df, period=10):
    momentum = df['Close'].diff(period)
    return momentum

def compute_stochastic_oscillator(df, window=14):
    lowest_low = df['Low'].rolling(window=window).min()
    highest_high = df['High'].rolling(window=window).max()
    stochastic_oscillator = ((df['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
    return stochastic_oscillator

def compute_gap(df):
    gap = df['Open'] - df['Close'].shift(1)
    return gap

def compute_volume_change(df):
    volume_change = df['Volume'].pct_change()
    return volume_change

def compute_ema(df, span):
    ema = df['Close'].ewm(span=span, adjust=False).mean()
    return ema

def compute_atr(df, window=14):
    true_range = compute_true_range(df)
    atr = true_range.rolling(window=window).mean()
    return atr

def compute_advanced_technical_indicators(df):
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(df['Close'])
    df['BB_High'], df['BB_Low'] = compute_bollinger_bands(df['Close'])
    df['BB_Width'] = df['BB_High'] - df['BB_Low']
    df['Historical_Volatility'] = compute_historical_volatility(df)
    df['Price_Volume'] = compute_price_volume(df)
    df['Momentum'] = compute_momentum(df)
    df['Stochastic_Oscillator'] = compute_stochastic_oscillator(df)
    df['Gap'] = compute_gap(df)
    df['Volume_Change'] = compute_volume_change(df)
    df['EMA_50'] = compute_ema(df, 50)
    df['EMA_200'] = compute_ema(df, 200)
    df['ATR'] = compute_atr(df)
    df['MFI'] = compute_mfi(df)
    df['CCI'] = compute_cci(df)

    df = df.ffill().bfill()
    
    return df

def update_daily_data(symbol, interval, api_key):
    data_dir = 'data/daily_data'
    original_data_dir = 'data/original_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(original_data_dir):
        os.makedirs(original_data_dir)
    
    files = os.listdir(data_dir)
    if not files:
        print("No existing data found. Fetching the last 120 days of data.")
        start_date = datetime.now() - timedelta(days=120)
        end_date = datetime.now()
    else:
        latest_file = max(files)
        latest_date_str = latest_file.split('_')[-1].split('.')[0]
        latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
        start_date = latest_date + timedelta(days=1)
        end_date = datetime.now()

    df = get_intraday_data(symbol, interval, api_key, start_date, end_date)
    df = df.sort_index()
    if df.empty:
        print("No new data fetched.")
        return

    print(f"Fetched data from {start_date} to {end_date}")
    print(f"Original data sample:\n{df.head()}")

    original_file_path = os.path.join(original_data_dir, f'{symbol}_intraday_{start_date.strftime("%Y-%m-%d")}_to_{end_date.strftime("%Y-%m-%d")}.csv')
    df.to_csv(original_file_path)
    print(f"Saved original data to {original_file_path}")

    df = df.sort_index()
    df['Previous_Close'] = df['Close'].shift(1)
    df['True_Range'] = compute_true_range(df)

    df = compute_advanced_technical_indicators(df)

    print("Completed feature engineering")
    print(f"Feature engineered data sample:\n{df.head()}")

    df.dropna(inplace=True)
    print(f"Data after dropping NA values:\n{df.head()}")
    print(f"Data shape after dropping NA values: {df.shape}")

    for date, group in df.groupby(df.index.date):
        output_file = os.path.join(data_dir, f'{symbol}_intraday_{date}.csv')
        group.to_csv(output_file)

    print(f"New data has been fetched and saved to {data_dir}")

update_daily_data(SYMBOL, INTERVAL, API_KEY)