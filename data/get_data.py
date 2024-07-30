import pandas as pd
import requests
import os
from datetime import datetime, timedelta
import time

API_KEY = 'PUDPSYYSPAF8IGTR'
SYMBOL = 'SPY'
INTERVAL = '1min'
API_CALLS_LIMIT = 5  # AlphaVantage每分钟最多允许5次API调用

def get_intraday_data(symbol, interval, api_key, start_date, end_date):
    data_frames = []
    current_date = end_date

    while current_date >= start_date:
        year_month = current_date.strftime("%Y-%m")
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}&outputsize=full&month={year_month}'
        print(f'Fetching data for {year_month}')
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

def compute_rsi(series, period):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

def compute_macd(series, slow=26, fast=12, signal=9):
    fast_ema = series.ewm(span=fast, min_periods=1, adjust=False).mean()
    slow_ema = series.ewm.span=slow, min_periods=1, adjust=False).mean()
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

def get_advanced_technical_indicators(df):
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(df['Close'])
    df['BB_High'], df['BB_Low'] = compute_bollinger_bands(df['Close'])
    df['BB_Width'] = df['BB_High'] - df['BB_Low']
    df['Historical_Volatility'] = df['Close'].rolling(window=21).std() * (252 ** 0.5)
    df['Price_Volume'] = df['Close'] * df['Volume']
    df['Momentum'] = df['Close'].diff(4)
    df['Stochastic_Oscillator'] = ((df['Close'] - df['Low'].rolling(window=14).min()) /
                                   (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())) * 100
    df['Gap'] = df['Open'] - df['Previous_Close']
    df['Volume_Change'] = df['Volume'].pct_change()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['ATR'] = df['True_Range'].rolling(window=14).mean()
    df['MFI'] = compute_mfi(df)
    df['CCI'] = compute_cci(df)
    df.dropna(inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

def compute_mfi(df):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow[df['Close'] > df['Close'].shift()]
    negative_flow = money_flow[df['Close'] < df['Close'].shift()]
    positive_flow = positive_flow.fillna(0)
    negative_flow = negative_flow.fillna(0)
    positive_mf = positive_flow.rolling(window=14).sum()
    negative_mf = negative_flow.rolling(window=14).sum()
    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    return mfi

def compute_cci(df, window=20):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    rolling_mean = typical_price.rolling(window=window).mean()
    rolling_std = typical_price.rolling(window=window).std()
    cci = (typical_price - rolling_mean) / (0.015 * rolling_std)
    return cci

# 获取数据
days_to_fetch = 90  # 用户可修改的参数，表示获取多少天的数据
start_date = None  # 可以设置为用户指定的起始日期，例如 datetime(2023, 1, 1)
end_date = None  # 可以设置为用户指定的结束日期，例如 datetime(2023, 6, 30)
df = get_full_data(SYMBOL, INTERVAL, API_KEY, start_date=start_date, end_date=end_date, days=days_to_fetch)

if not df.empty:
    # 重命名列并排序
    df = df.sort_index()

    # 数据清洗：删除重复数据
    df = df.drop_duplicates()

    # 添加Previous_Close列
    df['Previous_Close'] = df['Close'].shift(1)
    
    # 计算True_Range列
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    df['True_Range'] = high_low.combine(high_close, max).combine(low_close, max)

    # 进行复杂特征工程处理
    df = get_advanced_technical_indicators(df)

    # 创建保存数据的目录
    output_dir = './data/daily_data'
    os.makedirs(output_dir, exist_ok=True)

    # 按天切分数据并保存到独立的文件
    for date, group in df.groupby(df.index.date):
        output_file = os.path.join(output_dir, f'{SYMBOL}_intraday_{date}.csv')
        group.to_csv(output_file)

    print(f"Data has been fetched and saved to {output_dir}")
else:
    print("No data fetched. Please check the parameters or API limits.")
