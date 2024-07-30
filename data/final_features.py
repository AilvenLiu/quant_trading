import pandas as pd
import os

# Load the data
file_path = '/mnt/data/SPY_intraday_2024-07-29.csv'
df = pd.read_csv(file_path)
df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)

# Drop unnecessary features
columns_to_drop = ['Price_Change', 'SMA', 'MACD_Signal']
df.drop(columns=columns_to_drop, inplace=True)

# Add useful features
# ATR
high_low = df['High'] - df['Low']
high_close = (df['High'] - df['Close'].shift()).abs()
low_close = (df['Low'] - df['Close'].shift()).abs()
df['True_Range'] = high_low.combine(high_close, max).combine(low_close, max)
df['ATR'] = df['True_Range'].rolling(window=14).mean()

# MFI
typical_price = (df['High'] + df['Low'] + df['Close']) / 3
money_flow = typical_price * df['Volume']
positive_flow = money_flow[df['Close'] > df['Close'].shift()]
negative_flow = money_flow[df['Close'] < df['Close'].shift()]
positive_flow = positive_flow.fillna(0)
negative_flow = negative_flow.fillna(0)
positive_mf = positive_flow.rolling(window=14).sum()
negative_mf = negative_flow.rolling(window=14).sum()
mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
df['MFI'] = mfi

# CCI
typical_price = (df['High'] + df['Low'] + df['Close']) / 3
rolling_mean = typical_price.rolling(window=20).mean()
rolling_std = typical_price.rolling(window=20).std()
df['CCI'] = (typical_price - rolling_mean) / (0.015 * rolling_std)

# Remove any resulting NaN values after feature engineering
df.dropna(inplace=True)

# Save processed data
processed_file_path = '/mnt/data/SPY_intraday_2024-07-29_processed.csv'
df.to_csv(processed_file_path, index=False)

# Display the first few rows of the processed dataframe
df.head()
