import pandas as pd
import os
from datetime import datetime

SYMBOL = 'SPY'
ORIGINAL_DATA_PATH = 'data/original_data/SPY_intraday_2024-04-01_to_2024-07-30.csv'
DAILY_DATA_DIR = 'data/daily_data'

# 确保保存每日数据的目录存在
if not os.path.exists(DAILY_DATA_DIR):
    os.makedirs(DAILY_DATA_DIR)

def compute_rsi(series, period=6):
    delta = series.diff(1)
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

def compute_bollinger_bands(series, window=14, num_std_dev=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def compute_mfi(df, window=390*3):  # 3天滚动窗口的分钟数
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    positive_flow = money_flow.copy()
    negative_flow = money_flow.copy()
    
    positive_flow[typical_price <= typical_price.shift(1)] = 0
    negative_flow[typical_price >= typical_price.shift(1)] = 0
    
    positive_mf = positive_flow.rolling(window=window).sum()
    negative_mf = negative_flow.rolling(window=window).sum()
    
    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    return mfi

def compute_cci(df, window=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    rolling_mean = typical_price.rolling(window=window).mean()
    rolling_std = typical_price.rolling(window=window).std()
    cci = (typical_price - rolling_mean) / (0.015 * rolling_std)
    return cci

def compute_true_range(df):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = high_low.combine(high_close, max).combine(low_close, max)
    return true_range

def compute_historical_volatility(df, window=14):
    volatility = df['Close'].rolling(window=window).std() * (252 ** 0.5)
    return volatility

def compute_price_volume(df):
    price_volume = df['Close'] * df['Volume']
    return price_volume

def compute_momentum(df, period=3):
    momentum = df['Close'].diff(period)
    return momentum

def compute_stochastic_oscillator(df, window=14):
    lowest_low = df['Low'].rolling(window=window).min()
    highest_high = df['High'].rolling(window=window).max()
    stochastic_oscillator = ((df['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
    return stochastic_oscillator

def compute_gap(df):
    gap = df['Open'] - df['Previous_Close']
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

def get_advanced_technical_indicators(df):
    df['RSI'] = compute_rsi(df['Close'], 6)
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

    # 检查和处理生成的 NaN 值
    df = df.ffill().bfill()
    
    # 打印每个特征计算后的样本以确保数据有效
    print("RSI calculated")
    print(df[['RSI']].dropna().head())
    print("MACD calculated")
    print(df[['MACD', 'MACD_Signal', 'MACD_Hist']].dropna().head())
    print("Bollinger Bands calculated")
    print(df[['BB_High', 'BB_Low', 'BB_Width']].dropna().head())
    print("Historical Volatility calculated")
    print(df[['Historical_Volatility']].dropna().head())
    print("Price_Volume calculated")
    print(df[['Price_Volume']].dropna().head())
    print("Momentum calculated")
    print(df[['Momentum']].dropna().head())
    print("Stochastic Oscillator calculated")
    print(df[['Stochastic_Oscillator']].dropna().head())
    print("Gap calculated")
    print(df[['Gap']].dropna().head())
    print("Volume Change calculated")
    print(df[['Volume_Change']].dropna().head())
    print("EMA calculated")
    print(df[['EMA_50', 'EMA_200']].dropna().head())
    print("ATR calculated")
    print(df[['ATR']].dropna().head())
    print("MFI calculated")
    print(df[['MFI']].dropna().head())
    print("CCI calculated")
    print(df[['CCI']].dropna().head())

    return df

def process_and_save_data(file_path, output_dir):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    print(f"Original data sample:\n{df.head()}")

    # 数据按时间顺序排序
    df = df.sort_index()

    # 添加Previous_Close列
    df['Previous_Close'] = df['Close'].shift(1)
    
    # 计算True_Range列
    df['True_Range'] = compute_true_range(df)

    print(f"True Range calculation sample:\n{df['True_Range'].dropna().head()}")

    # 进行复杂特征工程处理
    df = get_advanced_technical_indicators(df)

    print("Completed feature engineering")
    print(f"Feature engineered data sample:\n{df.head()}")

    # 删除缺失值后的数据量
    df.dropna(inplace=True)
    print(f"Data after dropping NA values:\n{df.head()}")
    print(f"Data shape after dropping NA values: {df.shape}")

    # 按天切分并保存
    for date, group in df.groupby(df.index.date):
        output_file = os.path.join(output_dir, f'{SYMBOL}_intraday_{date}.csv')
        group.to_csv(output_file)

    print(f"Processed data has been saved to {output_dir}")

# 处理并保存数据
process_and_save_data(ORIGINAL_DATA_PATH, DAILY_DATA_DIR)