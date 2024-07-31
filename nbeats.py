import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from nbeats_pytorch.model import NBeatsNet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 数据加载和合并
data_dir = 'data/daily_data'
all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
data_list = []

for file in all_files:
    df = pd.read_csv(file, index_col='Datetime', parse_dates=True)
    data_list.append(df)

data = pd.concat(data_list)
data = data.dropna()

# 设置参数
input_window = 60  # 输入窗口大小（分钟）
output_window = 10  # 输出窗口大小（分钟）
train_size = 0.8  # 使用前80%的数据作为训练集
epochs = 100
batch_size = 32

# 数据预处理
def create_sequences(data, input_window, output_window):
    xs, ys = [], []
    for i in range(len(data) - input_window - output_window):
        x = data.iloc[i:(i + input_window)].values
        y = data.iloc[(i + input_window):(i + input_window + output_window)][['Open', 'High', 'Low', 'Close']].values
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 创建序列
features = data.drop(columns=['Open', 'High', 'Low', 'Close'])  # 使用除OHLC以外的特征进行预测
ohlc = data[['Open', 'High', 'Low', 'Close']]

x, y = create_sequences(data, input_window, output_window)

# 训练集和测试集划分
train_size = int(len(x) * train_size)
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# 将数据转换为 PyTorch 张量
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 构建 N-BEATSx 模型
model = NBeatsNet(device='cpu', stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), 
                  forecast_length=output_window * 4, backcast_length=input_window, thetas_dim=(4, 4), 
                  nb_blocks_per_stack=3, share_weights_in_stack=True, hidden_layer_units=512)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    forecast = model(x_train_tensor).squeeze()
    loss = criterion(forecast, y_train_tensor.view(-1, output_window * 4))
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 验证模型
model.eval()
with torch.no_grad():
    forecast = model(x_test_tensor).squeeze()
    loss = criterion(forecast, y_test_tensor.view(-1, output_window * 4))
    print(f'Validation Loss: {loss.item()}')

# 还原预测结果的形状
forecast = forecast.view(-1, output_window, 4).cpu().numpy()

# 可视化结果（仅以Close为例）
plt.figure(figsize=(15, 5))
plt.plot(y_test[:, -1, 3], label='True Close')  # 真实的Close值
plt.plot(forecast[:, -1, 3], label='Forecast Close')  # 预测的Close值
plt.legend()
plt.show()

# 提取 N-BEATSx 特征
model.eval()
with torch.no_grad():
    train_features = model.backcast(x_train_tensor).cpu().numpy()
    test_features = model.backcast(x_test_tensor).cpu().numpy()

# 将 N-BEATSx 特征作为 XGBoost 的输入
xgb_train = xgb.DMatrix(train_features, label=y_train[:, -1, 3])  # 使用Close作为标签
xgb_test = xgb.DMatrix(test_features, label=y_test[:, -1, 3])

# XGBoost 参数
params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100
}

# 训练 XGBoost 模型
xgb_model = xgb.train(params, xgb_train, num_boost_round=100)

# 预测
xgb_forecast = xgb_model.predict(xgb_test)

# 计算 XGBoost 的均方误差
xgb_mse = mean_squared_error(y_test[:, -1, 3], xgb_forecast)
print(f'XGBoost MSE: {xgb_mse}')

# 可视化 XGBoost 结果（仅以Close为例）
plt.figure(figsize=(15, 5))
plt.plot(y_test[:, -1, 3], label='True Close')
plt.plot(xgb_forecast, label='XGBoost Forecast Close')
plt.legend()
plt.show()