import pandas as pd
import os

# 复杂特征工程函数
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

# 数据目录
data_dir = 'data/daily_data'

# 遍历数据目录中的所有文件
for file_name in os.listdir(data_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path, parse_dates=True, index_col=0)

        # 进行复杂特征工程处理
        df['Previous_Close'] = df['Close'].shift(1)
        df = get_advanced_technical_indicators(df)

        # 保存处理后的数据
        df.to_csv(file_path)

print(f"Advanced feature engineering completed and saved for all daily data in {data_dir}")
