import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datasetsforecast.m4 import M4, M4Info

# --- 1. 定义Transformer模型 ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=8, num_encoder_layers=3, dim_feedforward=512, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        # src shape: [batch_size, sequence_length, input_dim]
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # 我们只关心序列最后的输出，用于预测下一个点
        output = self.decoder(output[:, -1, :])
        return output

# --- 2. 数据准备与训练函数 ---

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

def train_model(model, train_data, epochs=100, lr=0.001, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    model.train()
    print("--- 开始训练 Transformer 模型 ---")
    for epoch in range(epochs):
        total_loss = 0
        for seq, labels in train_loader:
            optimizer.zero_grad()
            
            # seq shape: [batch_size, input_window, 1]
            # label shape: [batch_size, 1, 1]
            seq = seq.float()
            # Squeeze the label tensor to match the model's output shape
            labels = labels.float().squeeze(-1)
            
            y_pred = model(seq)
            
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs} Loss: {avg_loss:.6f}')
    print("--- 训练完成 ---")


def evaluate_model(model, test_input_data, scaler, future_preds):
    model.eval()
    test_data_normalized = test_input_data.tolist()
    
    for _ in range(future_preds):
        seq = torch.FloatTensor(test_data_normalized[-input_window:]).view(1, -1, 1)
        with torch.no_grad():
            next_pred_normalized = model(seq).item()
            # Append as a list to maintain the structure (list of lists)
            test_data_normalized.append([next_pred_normalized])
            
    # Inverse transform the predictions
    # We skip the initial seed data and take only the predicted part
    predictions_normalized = np.array(test_data_normalized[input_window:]).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions_normalized).flatten()
    return predictions


# --- 3. 主执行逻辑 ---

if __name__ == '__main__':
    # --- 参数设定 ---
    group = 'Daily'
    input_window = 14
    
    # --- 数据加载 ---
    print(f"--- 正在加载M4 '{group}' 数据集 ---")
    Y_df, *_ = M4.load(directory='data', group=group)
    
    # M4.load returns a long format dataframe. 
    # Let's select the first time series to work with.
    series_id = Y_df['unique_id'].unique()[0]
    time_series_df = Y_df[Y_df['unique_id'] == series_id]
    
    horizon = M4Info[group].horizon
    
    # --- 数据分割与准备 ---
    train_series = time_series_df['y'].values[:-horizon]
    test_series_truth = time_series_df['y'].values[-horizon:]
    
    print(f"已加载序列 '{series_id}'，训练集长度: {len(train_series)}, 预测步长: {len(test_series_truth)}")

    # --- 数据标准化 ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_series.reshape(-1, 1))

    # --- 创建训练样本 ---
    train_inout_seq = create_inout_sequences(train_scaled, input_window)
    print(f"已创建 {len(train_inout_seq)} 个训练样本。")

    # --- 模型训练 ---
    model = TimeSeriesTransformer()
    train_model(model, train_inout_seq)
    
    # --- 模型评估 ---
    # 准备评估的输入数据：训练集的最后 input_window 个点 (normalized)
    test_inputs = train_scaled[-input_window:]
    predictions = evaluate_model(model, test_inputs, scaler, future_preds=horizon)
    
    # --- 结果打印 ---
    print("\n--- 预测结果 ---")
    print(f"真实未来值: \n{test_series_truth}")
    print(f"模型预测值: \n{predictions}")
    
    mse = np.mean((predictions - test_series_truth)**2)
    print(f"\n均方误差 (MSE): {mse:.2f}") 