import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

warnings.filterwarnings('ignore')

# --- 数据处理 ---
def get_time_series_data(dataset_type='min_daily_temps'):
    """从CSV加载并预处理时间序列数据"""
    if dataset_type == 'min_daily_temps':
        # 假设脚本与'data'目录在同一级别
        csv_path = os.path.join('data', 'min_daily_temps.csv')
        if not os.path.exists(csv_path):
             raise FileNotFoundError(f"找不到数据集 '{csv_path}'。请确保 'data' 文件夹和数据集存在。")
        df = pd.read_csv(csv_path)
        df.rename(columns={'Date': 'date', 'Temp': 'temp'}, inplace=True)
    else:
        raise ValueError('未知的数据集类型')
    
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# --- 模型定义 (从主实验中复制) ---
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.layers(x)

class EnhancedNN(nn.Module): # LSTM + Attention
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(EnhancedNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attention = nn.Linear(hidden_size, 1)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # PyTorch的LSTM期望输入形状为 (batch, seq, feature)，但我们的数据是 (batch, feature)
        # 我们需要在输入到LSTM前增加一个序列长度维度 (seq=1)
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        # lstm_out 形状: (batch, 1, hidden_size)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.bmm(lstm_out.transpose(1, 2), attn_weights).squeeze(2)
        return self.regressor(context)

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_encoder_layers=2):
        super(TransformerModel, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        # Transformer编码器也期望序列输入
        x = self.input_layer(x.unsqueeze(1)) # (batch, 1, d_model)
        x = self.transformer_encoder(x)
        x = self.output_layer(x.squeeze(1)) # (batch, 1)
        return x

# --- 模型训练 ---
def train_pytorch_model(model, X_train, y_train, X_test):
    """一个通用的PyTorch模型训练和预测函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(50): # 固定的训练周期
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_tensor = model(torch.FloatTensor(X_test).to(device))
    return preds_tensor.cpu().numpy().flatten()

# --- 特征工程 ---
def create_kitchen_sink_features(df, target_col):
    """创建一套'大而全'的特征集来模拟过度的特征工程"""
    df_out = df.copy()
    
    # 滞后特征
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df_out[f'lag_{lag}'] = df_out[target_col].shift(lag)
        
    # 差分特征
    for period in [1, 7, 30]:
        df_out[f'diff_{period}'] = df_out[target_col].diff(periods=period)
        
    # 滚动统计特征
    windows = [3, 7, 14, 30]
    for window in windows:
        rolling = df_out[target_col].rolling(window=window)
        df_out[f'rolling_mean_{window}'] = rolling.mean()
        df_out[f'rolling_std_{window}'] = rolling.std()
        df_out[f'rolling_min_{window}'] = rolling.min()
        df_out[f'rolling_max_{window}'] = rolling.max()
        df_out[f'rolling_skew_{window}'] = rolling.skew()
        df_out[f'rolling_kurt_{window}'] = rolling.kurt()
        
    # 指数加权移动平均
    for span in [7, 30]:
        df_out[f'ewm_span_{span}'] = df_out[target_col].ewm(span=span, adjust=False).mean()
        
    # 时间特征
    df_out['dayofweek'] = df_out['date'].dt.dayofweek
    df_out['month'] = df_out['date'].dt.month
    df_out['weekofyear'] = df_out['date'].dt.isocalendar().week.astype(int)
    df_out['quarter'] = df_out['date'].dt.quarter
    df_out['dayofyear'] = df_out['date'].dt.dayofyear
    df_out['is_weekend'] = (df_out['date'].dt.dayofweek >= 5).astype(int)
    
    # 傅里叶特征
    time_idx = (df_out['date'] - df_out['date'].min()).dt.days
    for period in [365.25, 30.5]:
        for k in range(1, 4): # 3阶傅里叶
            df_out[f'fourier_sin_{k}_{int(period)}'] = np.sin(2 * np.pi * k * time_idx / period)
            df_out[f'fourier_cos_{k}_{int(period)}'] = np.cos(2 * np.pi * k * time_idx / period)
            
    # 为所有基于目标变量衍生的特征进行移位，防止数据泄漏
    target_derived_cols = [col for col in df_out.columns if col not in df.columns and 'fourier' not in col and col not in ['dayofweek', 'month', 'weekofyear', 'quarter', 'dayofyear', 'is_weekend']]
    df_out[target_derived_cols] = df_out[target_derived_cols].shift(1)
    
    print(f"创建了 {len(df_out.columns) - len(df.columns)} 个新特征。")
    return df_out

# --- 核心评估逻辑 ---
def run_evaluation(df, target_col, models_def, feature_scenario_name):
    """在给定的数据集上运行所有模型的评估"""
    print(f"\n===== 评估场景: {feature_scenario_name} =====")
    
    # 核心修复: 不再使用 dropna()，而是填充NaN值
    # 先分离出特征和目标
    features = [col for col in df.columns if col not in ['date', target_col]]
    X = df[features]
    y = df[target_col]

    # 用0填充特征中的NaN。这是一种简单而稳健的处理方式，可以避免删除过多行。
    X = X.fillna(0)

    # 找出目标y中不是NaN的行索引
    valid_y_indices = y.notna()

    # 基于这些有效索引来过滤X和y，确保它们对齐且不含NaN
    X = X[valid_y_indices]
    y = y[valid_y_indices]

    if X.empty:
        print("警告: 没有可供评估的特征。")
        return {name: {'r2': float('nan')} for name in models_def}

    print(f"数据准备完毕，使用 {len(features)} 个特征进行训练。")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # 特征和目标值缩放
    scaler_X = MinMaxScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler()
    y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    results = {}
    for name, model_class in models_def.items():
        print(f"  -> 正在评估模型: {name}...")
        model = model_class(input_size=X.shape[1])
        
        preds_scaled = train_pytorch_model(model, X_train_s, y_train_s, X_test_s)
        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        
        r2 = r2_score(y_test, preds)
        results[name] = {'r2': r2}
        print(f"     - {name} R² 分数: {r2:.4f}")
    
    return results

def main():
    """主函数，执行完整的验证流程"""
    print("="*80)
    print("🚀 开始验证 '模型-特征不匹配' (Model-Feature Mismatch) 假设")
    print("="*80)
    
    # 1. 加载数据
    try:
        df_base = get_time_series_data('min_daily_temps')
        target_col = 'temp'
        print(f"成功加载数据集，包含 {len(df_base)} 条记录。")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

    # 2. 定义待评估的模型
    models_to_test = {
        'SimpleNN': SimpleNN,
        'EnhancedNN (LSTM+Attn)': EnhancedNN,
        'Transformer': TransformerModel
    }
    
    # 3. 实验一: 原始数据 (仅用 lag_1 作为最基础特征)
    df_raw = df_base.copy()
    df_raw['lag_1'] = df_raw[target_col].shift(1)
    results_raw = run_evaluation(df_raw, target_col, models_to_test, "原始数据 (仅 lag-1 特征)")
    
    # 4. 实验二: "大而全"特征集
    df_rich = create_kitchen_sink_features(df_base.copy(), target_col)
    results_rich = run_evaluation(df_rich, target_col, models_to_test, "'大而全'特征集")
    
    # 5. 打印总结报告
    print("\n\n" + "="*80)
    print("📊 假设验证总结报告")
    print("="*80)
    print(f"{'模型':<25} | {'R² (原始数据)':<20} | {'R² (大而全特征)':<20} | {'性能变化':<15}")
    print("-"*80)
    
    for model_name in models_to_test.keys():
        r2_raw = results_raw[model_name]['r2']
        r2_rich = results_rich[model_name]['r2']
        change = r2_rich - r2_raw
        
        change_str = f"{change:+.4f}"
        if change > 0.01:
            change_str += " (显著提升)"
        elif change < -0.01:
            change_str += " (显著下降)"
        else:
            change_str += " (无明显变化)"
            
        print(f"{model_name:<25} | {r2_raw:<20.4f} | {r2_rich:<20.4f} | {change_str:<15}")
        
    print("-"*80)
    print("\n结论:")
    print("  - SimpleNN: 在'大而全'特征集上表现是否提升？")
    print("  - EnhancedNN / Transformer: 在'大而全'特征集上表现是否下降？")
    print("如果上述问题的答案为'是'，则假设得到有力支持。")

if __name__ == "__main__":
    main() 