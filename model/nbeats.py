import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb
from torch.utils.data import DataLoader, Dataset

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_window, output_window):
        self.input_window = input_window
        self.output_window = output_window
        self.data = data.astype(np.float32)  # Ensure float32 dtype

    def __len__(self):
        return len(self.data) - self.input_window - self.output_window

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.input_window, :]
        y = self.data[idx + self.input_window:idx + self.input_window + self.output_window, 3]  # 仅使用 Close 作为目标
        return x, y

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
                nn.Dropout(0.2),  # 添加 Dropout
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU(),
                nn.Dropout(0.2),  # 添加 Dropout
                nn.Linear(hidden_units, input_dim * backcast_length + forecast_length)
            )

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten the input
            return self.fc(x)

    def forward(self, x):
        residual = x
        forecast = torch.zeros((x.size(0), self.forecast_length), device=x.device, dtype=torch.float32)  # Ensure float32
        for stack in self.stacks:
            for block in stack:
                block_forecast = block(residual)
                backcast_size = self.backcast_length * self.input_dim
                forecast_size = self.forecast_length
                backcast = block_forecast[:, :backcast_size]
                forecast_delta = block_forecast[:, -forecast_size:]
                residual = residual - backcast.view(x.size(0), self.backcast_length, self.input_dim)
                forecast += forecast_delta
        return forecast

def create_sequences(data, input_window, output_window):
    xs, ys = [], []
    for i in range(len(data) - input_window - output_window):
        x = data[i:(i + input_window), :].astype(np.float32)  # Ensure float32 dtype
        y = data[(i + input_window):(i + input_window + output_window), 3].astype(np.float32)  # 仅使用 Close 作为目标
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_nbeatsx_model(train_loader, input_dim, output_window, input_window, hidden_units, epochs=100):
    # 初始化 N-BEATSx 模型
    model = NBeatsx(
        input_dim=input_dim,
        forecast_length=output_window,
        backcast_length=input_window,
        hidden_units=hidden_units,
        stacks=3,
        blocks_per_stack=3
    ).to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练 N-BEATSx 模型
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            forecast = model(x_batch)
            loss = criterion(forecast, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {epoch_loss / len(train_loader)}')
    
    return model

def evaluate_model(model, data_loader, output_window):
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            forecast = model(x_batch)
            predictions.extend(forecast.cpu().numpy())
            true_values.extend(y_batch.cpu().numpy())
    return np.array(predictions), np.array(true_values)

def visualize_predictions(predictions, true_values, file_name, output_dir):
    num_samples = len(predictions)
    step_size = 120  # 每个图像包含的时间步长
    for start in range(0, num_samples, step_size):
        end = start + step_size
        plt.figure(figsize=(15, 5))
        plt.plot(true_values[start:end], label='True Close', color='blue')
        plt.plot(predictions[start:end], label='Predicted Close', color='orange')
        plt.title(f'{file_name} - Sample {start} to {end}')
        plt.xlabel('Time Step')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        output_file = os.path.join(output_dir, f'{file_name}_samples_{start}_to_{end}.png')
        plt.savefig(output_file)
        plt.close()

def main():
    # 加载数据
    data_dir = '../data/daily_data'
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

    # 计算 80% 的文件用于训练，20% 用于测试
    split_index = int(len(all_files) * 0.8)
    train_files = all_files[:split_index]
    test_files = all_files[split_index:]

    input_window = 60  # 输入窗口大小（分钟）
    output_window = 10  # 输出窗口大小（分钟）
    hidden_units = 256  # 隐藏层单元数量

    # 训练模型
    train_data_list = []
    for file in train_files:
        df = pd.read_csv(file, index_col='Datetime', parse_dates=True)
        data_scaled = StandardScaler().fit_transform(df.dropna().astype(np.float32))  # Ensure float32 dtype
        train_data_list.append(data_scaled)

    train_data = np.concatenate(train_data_list, axis=0)
    train_dataset = TimeSeriesDataset(train_data, input_window, output_window)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 模型训练
    model = train_nbeatsx_model(train_loader, input_dim=train_data.shape[1], output_window=output_window, input_window=input_window, hidden_units=hidden_units)

    # 验证模型并保存结果
    for file in test_files:
        df = pd.read_csv(file, index_col='Datetime', parse_dates=True)
        data_scaled = StandardScaler().fit_transform(df.dropna().astype(np.float32))  # Ensure float32 dtype
        test_dataset = TimeSeriesDataset(data_scaled, input_window, output_window)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 预测和评估
        predictions, true_values = evaluate_model(model, test_loader, output_window)

        # 创建文件夹保存预测结果的图像
        output_dir = os.path.join('predictions', os.path.basename(file).replace('.csv', ''))
        os.makedirs(output_dir, exist_ok=True)

        # 可视化预测结果
        visualize_predictions(predictions, true_values, os.path.basename(file), output_dir)

if __name__ == '__main__':
    main()

