import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
import os
from probes.quantum_probe import QuantumDualStreamProbe

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data(data_path, sequence_length=7):
    # 读取数据
    df = pd.read_csv(data_path)
    df['日期'] = pd.to_datetime(df['日期'])
    df.set_index('日期', inplace=True)
    
    # 数据标准化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['成交商品件数']])
    
    # 创建序列数据
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)].flatten())  # 将序列展平
        y.append(scaled_data[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch, torch.zeros(X_batch.size(0), 1).long().to(device))  # 使用零向量作为定性特征
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch, torch.zeros(X_batch.size(0), 1).long().to(device))
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            
            predictions.extend(y_pred.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())
    
    return total_loss / len(test_loader), np.array(predictions), np.array(actuals)

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 准备数据
    X_train, X_test, y_train, y_test, scaler = prepare_data('data/total_cleaned.csv')
    
    # 创建数据加载器
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    input_size = X_train.shape[1]  # 修改为展平后的特征维度
    model = QuantumDualStreamProbe(
        quant_input_size=input_size,
        vocab_size=1,  # 由于我们使用零向量作为定性特征
        qual_embed_dim=16,
        quant_embed_dim=48
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    n_epochs = 100
    best_test_loss = float('inf')
    best_model_state = None
    
    for epoch in range(n_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, predictions, actuals = evaluate_model(model, test_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{n_epochs}:')
        print(f'Training Loss: {train_loss:.6f}')
        print(f'Testing Loss: {test_loss:.6f}')
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict().copy()
    
    # 保存最佳模型
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(best_model_state, 'saved_models/quantum_probe_new.pth')
    
    # 计算评估指标
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)
    
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    # 保存结果
    results = {
        'model': 'QuantumDualStreamProbe',
        'dataset': 'total_cleaned',
        'metrics': {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape)
        },
        'best_test_loss': float(best_test_loss),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results_QuantumProbe_new.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main() 