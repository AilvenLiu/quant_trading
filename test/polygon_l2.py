import pandas as pd
import requests
import os
from datetime import datetime
import time

API_KEY = 'QOWvSpapnH75SiUv_Xhfm9_BEDx8vFSy'
SYMBOL = 'SPY'
DATA_DIR = 'data/daily_data_l2'
SAVE_INTERVAL = 10  # 保存间隔（分钟）

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_order_book_data(symbol, api_key):
    url = f'https://api.polygon.io/v3/snapshot/locale/us/markets/stocks/tickers/{symbol}/orderbook?apiKey={api_key}'
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
            return combined
        else:
            print(f"No data for symbol {symbol}")
    else:
        print(f"HTTP Error: {response.status_code}")
    return pd.DataFrame()

def save_data(df, date_str):
    file_path = os.path.join(DATA_DIR, f'{SYMBOL}_orderbook_{date_str}.csv')
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False)
    else:
        df.to_csv(file_path)

def main():
    date_str = datetime.now().strftime("%Y-%m-%d")
    combined_data = pd.DataFrame()
    save_counter = 0

    while True:
        order_book_data = get_order_book_data(SYMBOL, API_KEY)
        if not order_book_data.empty:
            print(order_book_data.head())
            combined_data = pd.concat([combined_data, order_book_data])

        save_counter += 1
        if save_counter >= SAVE_INTERVAL:
            save_data(combined_data, date_str)
            combined_data = pd.DataFrame()  # 重置数据
            save_counter = 0

        time.sleep(60)  # 每分钟获取一次数据

if __name__ == "__main__":
    main()