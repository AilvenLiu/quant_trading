import pandas as pd
import os

# 读取已保存的完整数据
data_path = 'data/SPY_intraday_cleaned.csv'
df = pd.read_csv(data_path, parse_dates=True, index_col=0)

# 创建一个目录来保存切分后的文件
output_dir = 'data/daily_data'
os.makedirs(output_dir, exist_ok=True)

# 按天切分数据并保存到独立的文件
for date, group in df.groupby(df.index.date):
    output_file = os.path.join(output_dir, f'SPY_intraday_{date}.csv')
    group.to_csv(output_file)

print(f"Data has been split and saved to {output_dir}")
