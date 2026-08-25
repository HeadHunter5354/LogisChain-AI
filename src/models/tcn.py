import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

class Chomp1d(nn.Module):
    """Ensures causality by trimming padding from the future time steps"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.forecast_head = nn.Linear(num_channels[-1], 30) # Multi-horizon 30-day ahead forecast

    def forward(self, x):
        y = self.network(x)
        out = self.forecast_head(y[:, :, -1])
        return out

def create_sequences(data, input_seq_len=60, forecast_horizon=30):
    X, y = [], []
    for i in range(len(data) - input_seq_len - forecast_horizon + 1):
        X.append(data[i : i + input_seq_len])
        y.append(data[i + input_seq_len : i + input_seq_len + forecast_horizon, 0])
    return np.array(X), np.array(y)

def train_tcn_pipeline():
    print("Loading port and freight rate time-series...")
    port_df = pd.read_csv("data/raw/port_throughput.csv")
    
    # Feature columns: Throughput (Target), Berth Util, Dwell Days, Spot Rate
    series_cols = ["throughput_teu", "berth_utilization", "avg_dwell_time_days", "container_spot_rate"]
    raw_values = port_df[series_cols].values
    
    # Normalization
    mean = raw_values.mean(axis=0)
    std = raw_values.std(axis=0) + 1e-7
    scaled_values = (raw_values - mean) / std

    input_len, horizon = 60, 30
    X, y = create_sequences(scaled_values, input_seq_len=input_len, forecast_horizon=horizon)

    # Train / Test split on time-series (strict temporal order)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Shape adjustment: (Batch, Channels, Seq_Len)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)

    # Model definition (dilations: 1, 2, 4, 8)
    model = TemporalConvNet(num_inputs=len(series_cols), num_channels=[32, 64, 64, 32], kernel_size=3, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(1, 121):
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_preds_scaled = model(X_test_t).numpy()
    
    # Rescale throughput to true scale
    true_y = y_test * std[0] + mean[0]
    pred_y = test_preds_scaled * std[0] + mean[0]

    mape = mean_absolute_percentage_error(true_y, pred_y) * 100.0
    rmse = np.sqrt(mean_squared_error(true_y, pred_y))

    print("=" * 50)
    print("TCN TIME-SERIES FORECAST RESULTS")
    print(f"Horizon:          30 Days Ahead")
    print(f"Throughput MAPE:  {mape:.2f}% (Target: <12%)")
    print(f"Throughput RMSE:  {rmse:.2f} TEU")
    print("=" * 50)

    # Save trained forecasts
    np.save("data/processed/tcn_30d_forecast.npy", pred_y)
    print("30-Day Ahead Forecasts saved to data/processed/tcn_30d_forecast.npy")

if __name__ == "__main__":
    train_tcn_pipeline()