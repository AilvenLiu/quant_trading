import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_window, output_window):
        self.input_window = input_window
        self.output_window = output_window
        self.data = data.astype(np.float32)

    def __len__(self):
        return len(self.data) - self.input_window - self.output_window

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.input_window, :]
        y = self.data[idx + self.input_window:idx + self.input_window + self.output_window, 3]  # 仅使用 Close 作为目标
        return x, y

# 模型定义
class MarketModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, output_window):
        super(MarketModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, d_model, num_layers=2, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=0.2),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, output_window)

    def forward(self, x):
        # LSTM
        x, _ = self.lstm(x)
        
        # Transformer expects input [seq_len, batch_size, d_model], transpose dimensions
        x = x.permute(1, 0, 2)
        
        # Transformer
        x = self.transformer(x)
        
        # Get the last output from the sequence
        x = x[-1, :, :]
        
        # Fully connected layer
        x = self.fc(x)
        return x

def create_sequences(data, input_window, output_window, step_size):
    xs, ys = [], []
    for i in range(0, len(data) - input_window - output_window + 1, step_size):
        x = data[i:(i + input_window), :].astype(np.float32)
        y = data[(i + input_window):(i + input_window + output_window), 3].astype(np.float32)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model(train_loader, input_dim, d_model, num_heads, num_layers, output_window, input_window, epochs=300):
    # 初始化 MarketModel 模型
    model = MarketModel(input_dim=input_dim, d_model=d_model, num_heads=num_heads, num_layers=num_layers, output_window=output_window).to(device)
    
    # 损失函数和优化器
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    # 模型训练
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
        
        scheduler.step(epoch_loss / len(train_loader))
        
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

def visualize_predictions_gif(predictions, true_values, file_name, output_dir, step_size=1, output_window=10):
    assert len(predictions) == len(true_values), "Predictions and true values must have the same length."

    num_samples = len(predictions)
    fig, ax = plt.subplots(figsize=(15, 5))

    line_true, = ax.plot([], [], label='True Close', color='blue', lw=1.5)
    line_pred, = ax.plot([], [], label='Predicted Close', color='orange', lw=1.5)

    ax.set_title(f"{file_name}")
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Close Price')
    ax.legend()
    ax.grid(True)

    def init():
        line_true.set_data([], [])
        line_pred.set_data([], [])
        return line_true, line_pred

    def update(frame):
        start = 0
        end = frame + output_window
        if end > num_samples:
            end = num_samples

        x = np.arange(start, end)
        y_true = true_values[start:end, -1]
        y_pred = predictions[start:end, -1]

        line_true.set_data(x, y_true)
        line_pred.set_data(x, y_pred)

        ax.set_xlim(0, end)
        y_min = min(np.min(y_true), np.min(y_pred)) - 5
        y_max = max(np.max(y_true), np.max(y_pred)) + 5
        ax.set_ylim(y_min, y_max)

        return line_true, line_pred

    ani = FuncAnimation(fig, update, frames=np.arange(0, num_samples - output_window, step_size),
                        init_func=init, blit=True, repeat=False)
    
    gif_file = os.path.join(output_dir, f'{file_name}.gif')
    ani.save(gif_file, writer=PillowWriter(fps=5))
    plt.close()

def main():
    data_dir = 'data/daily_data'
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

    split_index = int(len(all_files) * 0.8)
    train_files = all_files[:split_index]
    test_files = all_files[split_index:]

    input_window = 60
    output_window = 10
    hidden_units = 256  # 此行无效，因为 `hidden_units` 不在 `train_model` 使用
    step_size = 1
    d_model = 64
    num_heads = 4
    num_layers = 2

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    train_data_list = []
    all_close_prices = []
    
    for file in train_files:
        df = pd.read_csv(file, index_col='Datetime', parse_dates=True)
        all_close_prices.append(df['Close'].values)
        data_scaled = feature_scaler.fit_transform(df.dropna().astype(np.float32))
        train_data_list.append(data_scaled)

    all_close_prices = np.concatenate(all_close_prices, axis=0).reshape(-1, 1)
    target_scaler.fit(all_close_prices)

    train_data = np.concatenate(train_data_list, axis=0)
    train_dataset = TimeSeriesDataset(train_data, input_window, output_window)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)

    model = train_model(
        train_loader,
        input_dim=train_data.shape[1],
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        output_window=output_window,
        input_window=input_window
    )

    for file in test_files:
        df = pd.read_csv(file, index_col='Datetime', parse_dates=True)
        data_scaled = feature_scaler.transform(df.dropna().astype(np.float32))
        test_dataset = TimeSeriesDataset(data_scaled, input_window, output_window)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

        predictions, true_values = evaluate_model(model, test_loader, output_window)

        predictions = target_scaler.inverse_transform(predictions)
        true_values = target_scaler.inverse_transform(true_values)

        output_dir = os.path.join('model', 'predictions', os.path.basename(file).replace('.csv', ''))
        os.makedirs(output_dir, exist_ok=True)

        visualize_predictions_gif(predictions, true_values, os.path.basename(file), output_dir, step_size=step_size, output_window=output_window)

if __name__ == '__main__':
    main()