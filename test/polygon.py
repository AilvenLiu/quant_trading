# Polygon.io API 密钥
# API_KEY = 'QOWvSpapnH75SiUv_Xhfm9_BEDx8vFSy'

import pandas as pd
import requests
import os
from datetime import datetime, timedelta
import time
import numpy as np

API_KEY = 'QOWvSpapnH75SiUv_Xhfm9_BEDx8vFSy'
SYMBOL = 'SPY'
INTERVAL = '1min'
API_CALLS_LIMIT = 5  # 每分钟最多允许5次API调用

def get_intraday_data(symbol, api_key, start_date, end_date):
    data_frames = []
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{date_str}/{date_str}?apiKey={api_key}'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                df = pd.DataFrame(data['results'])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df.index.name = 'Datetime'
                df = df.rename(columns={
                    'o': 'Open',
                    'h': 'High',
                    'l': 'Low',
                    'c': 'Close',
                    'v': 'Volume'
                })
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                data_frames.append(df)
            else:
                print(f"No data for {date_str}")
        else:
            print(f"HTTP Error: {response.status_code}")

        current_date = current_date + timedelta(days=1)
        time.sleep(60 / API_CALLS_LIMIT)  # Respect API rate limit

    if data_frames:
        df = pd.concat(data_frames)
        df = df.sort_index()
        return df
    else:
        return pd.DataFrame()

def get_level2_data(symbol, api_key, start_date, end_date):
    data_frames = []
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        url = f'https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}/orderbook?apiKey={api_key}'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'ticker' in data:
                bids = pd.DataFrame(data['ticker']['bids'])
                asks = pd.DataFrame(data['ticker']['asks'])
                bids['timestamp'] = pd.to_datetime(bids['t'], unit='ms')
                asks['timestamp'] = pd.to_datetime(asks['t'], unit='ms')
                bids.set_index('timestamp', inplace=True)
                asks.set_index('timestamp', inplace=True)
                bids.index.name = 'Datetime'
                asks.index.name = 'Datetime'
                bids = bids[['p', 's']]
                asks = asks[['p', 's']]
                bids.columns = ['Bid_Price', 'Bid_Size']
                asks.columns = ['Ask_Price', 'Ask_Size']
                combined = bids.join(asks, how='outer')
                combined = combined.ffill().bfill()
                data_frames.append(combined)
            else:
                print(f"No data for {date_str}")
        else:
            print(f"HTTP Error: {response.status_code}")

        current_date = current_date + timedelta(days=1)
        time.sleep(60 / API_CALLS_LIMIT)  # Respect API rate limit

    if data_frames:
        df = pd.concat(data_frames)
        df = df.sort_index()
        return df
    else:
        return pd.DataFrame()

def compute_true_range(df):
    df['High-Low'] = df['High'] - df['Low']
    df['High-Previous_Close'] = np.abs(df['High'] - df['Previous_Close'])
    df['Low-Previous_Close'] = np.abs(df['Low'] - df['Previous_Close'])
    df['True_Range'] = df[['High-Low', 'High-Previous_Close', 'Low-Previous_Close']].max(axis=1)
    return df

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
    df['EMA_50'] = compute_ema(df['Close'], 50)
    df['EMA_200'] = compute_ema(df['Close'], 200)
    df['ATR'] = compute_atr(df)
    df['MFI'] = compute_mfi(df)
    df['CCI'] = compute_cci(df)
    df['Consolidation_Range'] = compute_consolidation_range(df)
    df['Keltner_High'], df['Keltner_Low'] = compute_keltner_channel(df)
    df['Donchian_High'], df['Donchian_Low'] = compute_donchian_channel(df)
    df['OBV'] = compute_obv(df)
    df['CMF'] = compute_cmf(df)
    df = df.ffill().bfill()
    
    return df

# 各种技术指标计算函数
def compute_rsi(series, window=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - signal_line
    return macd, signal_line, macd_hist

def compute_bollinger_bands(series, window=20):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

def compute_historical_volatility(df, window=30):
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    volatility = log_returns.rolling(window).std() * np.sqrt(252)  # Annualize
    return volatility

def compute_price_volume(df):
    return df['Close'] * df['Volume']

def compute_momentum(df, window=10):
    return df['Close'].diff(window)

def compute_stochastic_oscillator(df, window=14):
    low_min = df['Low'].rolling(window=window).min()
    high_max = df['High'].rolling(window=window).max()
    k = 100 * (df['Close'] - low_min) / (high_max - low_min)
    return k

def compute_gap(df):
    df['Gap'] = df['Open'] - df['Previous_Close']
    return df['Gap']

def compute_volume_change(df):
    return df['Volume'].pct_change()

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Previous_Close'])
    low_close = np.abs(df['Low'] - df['Previous_Close'])
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

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
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    return cci

def compute_consolidation_range(df, window=30):
    rolling_high = df['High'].rolling(window=window).max()
    rolling_low = df['Low'].rolling(window=window).min()
    consolidation_range = rolling_high - rolling_low
    return consolidation_range

def compute_keltner_channel(df, window=20):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    kelter_middle = typical_price.rolling(window).mean()
    atr = compute_atr(df, window)
    kelter_high = kelter_middle + (2 * atr)
    kelter_low = kelter_middle - (2 * atr)
    return kelter_high, kelter_low

def compute_donchian_channel(df, window=20):
    donchian_high = df['High'].rolling(window).max()
    donchian_low = df['Low'].rolling(window).min()
    return donchian_high, donchian_low

def compute_obv(df):
    obv = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    return obv

def compute_cmf(df, window=20):
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    cmf = mfv.rolling(window).sum() / df['Volume'].rolling(window).sum()
    return cmf

def update_daily_data(symbol, api_key):
    data_dir = 'data/daily_data'
    original_data_dir = 'data/original_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(original_data_dir):
        os.makedirs(original_data_dir)
    
    start_date = datetime.now() - timedelta(days=120)
    end_date = datetime.now()

    df = get_intraday_data(symbol, api_key, start_date, end_date)
    if df.empty:
        print("No new data fetched.")
        return

    print(f"Fetched data from {start_date} to {end_date}")
    print(f"Original data sample:\n{df.head()}")

    original_file_path = os.path.join(original_data_dir, f'{symbol}_intraday_{start_date.strftime("%Y-%m-%d")}_to_{end_date.strftime("%Y-%m-%d")}.csv')
    df.to_csv(original_file_path)
    print(f"Saved original data to {original_file_path}")

    df['Previous_Close'] = df['Close'].shift(1)
    df = compute_true_range(df)
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

update_daily_data(SYMBOL, API_KEY)
