import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载和合并
data_dir = 'data/daily_data'
all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
data_list = []

for file in all_files:
    df = pd.read_csv(file, index_col='Datetime', parse_dates=True)
    data_list.append(df)

data = pd.concat(data_list)
data = data.dropna()

# 检查数据维度和列名
print("数据维度:", data.shape)
print("数据列名:", data.columns)

# 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 设置参数
input_window = 60  # 输入窗口大小（分钟）
output_window = 10  # 输出窗口大小（分钟）
train_size_ratio = 0.8  # 使用前80%的数据作为训练集
epochs = 100
batch_size = 32

# 数据预处理
def create_sequences(data, input_window, output_window):
    xs, ys = [], []
    for i in range(len(data) - input_window - output_window):
        x = data[i:(i + input_window), :]  # 多变量输入
        y = data[(i + input_window):(i + input_window + output_window), 3]  # 仅使用 Close 作为目标
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 创建序列
x, y = create_sequences(data_scaled, input_window, output_window)

# 训练集和测试集划分
train_size = int(len(x) * train_size_ratio)
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# 将数据转换为 PyTorch 张量
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# 自定义的 N-BEATSx 模型
class NBeatsx(nn.Module):
    def __init__(self, input_dim, forecast_length, backcast_length, hidden_units, stacks, blocks_per_stack):
        super(NBeatsx, self).__init__()
        self.input_dim = input_dim
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_units = hidden_units
        self.stacks = nn.ModuleList([
            nn.ModuleList([
                self.Block(input_dim, forecast_length, backcast_length, hidden_units)
                for _ in range(blocks_per_stack)
            ])
            for _ in range(stacks)
        ])

    class Block(nn.Module):
        def __init__(self, input_dim, forecast_length, backcast_length, hidden_units):
            super(NBeatsx.Block, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim * backcast_length, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, input_dim * backcast_length + forecast_length)  # 预测 Close
            )

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten the input
            return self.fc(x)

    def forward(self, x):
        residual = x
        forecast = torch.zeros((x.size(0), self.forecast_length), device=x.device)  # 调整 forecast 维度
        for stack in self.stacks:
            for block in stack:
                block_forecast = block(residual)
                backcast_size = self.backcast_length * self.input_dim
                forecast_size = self.forecast_length  # 仅预测 Close
                # 计算 backcast 和 forecast_delta
                backcast = block_forecast[:, :backcast_size]
                forecast_delta = block_forecast[:, -forecast_size:]
                # 重新调整 backcast 的维度
                residual = residual - backcast.view(x.size(0), self.backcast_length, self.input_dim)
                forecast += forecast_delta
        return forecast

# 初始化 N-BEATSx 模型
model = NBeatsx(
    input_dim=x_train_tensor.shape[-1],  # 输入特征的数量
    forecast_length=output_window,  # 输出窗口大小
    backcast_length=input_window,  # 输入窗口大小
    hidden_units=512,  # 隐藏层单元数量
    stacks=3,  # 堆栈数量增加以提高表达能力
    blocks_per_stack=3  # 每个堆栈中的块数量
).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练 N-BEATSx 模型
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    forecast = model(x_train_tensor)
    loss = criterion(forecast, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 提取 N-BEATSx 特征
model.eval()
with torch.no_grad():
    train_features = model(x_train_tensor).cpu().numpy()
    test_features = model(x_test_tensor).cpu().numpy()

# 将 N-BEATSx 特征作为 XGBoost 的输入
xgb_train = xgb.DMatrix(train_features, label=y_train[:, -1])  # 仅预测 Close
xgb_test = xgb.DMatrix(test_features, label=y_test[:, -1])

# XGBoost 参数
params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.05,  # 降低学习率以提高稳定性
    'n_estimators': 100
}

# 训练 XGBoost 模型
xgb_model = xgb.train(params, xgb_train, num_boost_round=100)

# 预测
xgb_forecast = xgb_model.predict(xgb_test)

# 计算 XGBoost 的均方误差
xgb_mse = mean_squared_error(y_test[:, -1], xgb_forecast)
print(f'XGBoost MSE: {xgb_mse}')

# 将预测结果反标准化
xgb_forecast_inverse = scaler.inverse_transform(np.c_[np.zeros((xgb_forecast.shape[0], 34)), xgb_forecast])[:, -1]
y_test_inverse = scaler.inverse_transform(np.c_[np.zeros((y_test.shape[0], 34)), y_test[:, -1]])[:, -1]

# 可视化 XGBoost 结果（仅以Close为例）
plt.figure(figsize=(15, 5))
plt.plot(y_test_inverse, label='True Close')
plt.plot(xgb_forecast_inverse, label='XGBoost Forecast Close')
plt.legend()
plt.show()