import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import lr_scheduler
import matplotlib.patches as mpatches
import torch.nn.functional as F
from statsmodels.tsa.arima.model import ARIMA

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_advanced_features(df, target_col='成交商品件数'):
    """创建高级特征"""
    df_copy = df.copy()
    df_copy['日期'] = pd.to_datetime(df_copy['日期'])
    
    # 基础时间特征
    df_copy['dayofweek'] = df_copy['日期'].dt.dayofweek
    df_copy['is_weekend'] = df_copy['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df_copy['month'] = df_copy['日期'].dt.month
    df_copy['day'] = df_copy['日期'].dt.day
    
    # 周期性特征
    df_copy['sin_day'] = np.sin(2 * np.pi * df_copy['日期'].dt.day / 31)
    df_copy['cos_day'] = np.cos(2 * np.pi * df_copy['日期'].dt.day / 31)
    df_copy['sin_month'] = np.sin(2 * np.pi * df_copy['日期'].dt.month / 12)
    df_copy['cos_month'] = np.cos(2 * np.pi * df_copy['日期'].dt.month / 12)
    
    # 滞后特征
    for i in range(1, 15):
        df_copy[f'lag_{i}'] = df_copy[target_col].shift(i)
    
    # 移动平均特征 (修复数据泄露：增加 .shift(1))
    for window in [7, 14, 30]:
        df_copy[f'rolling_mean_{window}'] = df_copy[target_col].rolling(window=window, min_periods=1).mean().shift(1)
        df_copy[f'rolling_std_{window}'] = df_copy[target_col].rolling(window=window, min_periods=1).std().shift(1)
    
    # 差分特征 (修复数据泄露：增加 .shift(1))
    df_copy['diff_1'] = df_copy[target_col].diff().shift(1)
    df_copy['diff_7'] = df_copy[target_col].diff(7).shift(1)
    
    # 趋势特征 (修复数据泄露：增加 .shift(1))
    df_copy['trend'] = df_copy[target_col].rolling(window=7, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    ).shift(1)
    
    # 处理缺失值 (修复数据泄露：使用dropna()代替bfill)
    df_copy.dropna(inplace=True)
    
    return df_copy

def create_sequences(X, y, sequence_length=14):
    """为序列模型创建时间序列样本"""
    xs, ys = [], []
    for i in range(len(X) - sequence_length):
        xs.append(X[i:(i + sequence_length)])
        ys.append(y[i + sequence_length])
    return np.array(xs), np.array(ys)

def get_feature_groups(feature_names):
    """根据特征名称列表将特征分组"""
    groups = {
        'time': ['dayofweek', 'is_weekend', 'month', 'day'],
        'cyclical': ['sin_day', 'cos_day', 'sin_month', 'cos_month'],
        'lag': [],
        'rolling_mean': [],
        'rolling_std': [],
        'diff_trend': []
    }
    # For dynamically generated features
    for name in feature_names:
        if 'lag_' in name:
            groups['lag'].append(name)
        elif 'rolling_mean' in name:
            groups['rolling_mean'].append(name)
        elif 'rolling_std' in name:
            groups['rolling_std'].append(name)
        elif 'diff' in name or 'trend' in name:
            groups['diff_trend'].append(name)
    return groups

def filter_features(df_features, excluded_groups=None):
    """根据要排除的特征组来筛选特征"""
    target_col = '成交商品件数'
    base_feature_cols = [c for c in df_features.columns if c not in ['日期', target_col]]
    
    if not excluded_groups:
        return base_feature_cols, df_features

    print(f"  正在排除特征组: {excluded_groups}")
    
    all_feature_groups = get_feature_groups(base_feature_cols)
    
    features_to_drop = []
    for group_name in excluded_groups:
        features_to_drop.extend(all_feature_groups.get(group_name, []))
    
    final_feature_cols = [f for f in base_feature_cols if f not in features_to_drop]
    
    print(f"  原始特征数量: {len(base_feature_cols)}, 移除后: {len(final_feature_cols)}")
    
    return final_feature_cols, df_features

def prepare_data(df, sequence_length, train_ratio=0.7, val_ratio=0.15, excluded_features=None):
    """
    数据准备的总函数，包括特征创建、筛选、缩放和分割。
    现在统一处理数据，返回一个包含flat和sequence两种格式的字典。
    """
    # 1. 创建高级特征
    target_col = '成交商品件数'
    df_features = create_advanced_features(df, target_col=target_col)
    
    # 2. 特征筛选
    feature_cols, df_features = filter_features(df_features, excluded_features)
    
    X = df_features[feature_cols].values
    y = df_features[target_col].values
    
    # 3. 数据分割点
    total_size = len(X)
    train_end = int(total_size * train_ratio)
    val_end = train_end + int(total_size * val_ratio)
    
    # 4. 数据缩放 (仅用训练集拟合)
    X_scaler = RobustScaler().fit(X[:train_end])
    y_scaler = RobustScaler().fit(y[:train_end].reshape(-1, 1))
    
    X_scaled = X_scaler.transform(X)
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).flatten()
    
    # 5. 分割完整数据集
    X_train_scaled, y_train_scaled = X_scaled[:train_end], y_scaled[:train_end]
    X_val_scaled, y_val_scaled = X_scaled[train_end:val_end], y_scaled[train_end:val_end]
    X_test_scaled, y_test_scaled = X_scaled[val_end:], y_scaled[val_end:]
    
    # --- 为不同类型的模型创建对齐的数据 ---
    
    # 6. "Flat" 数据 (用于XGBoost, RF, SimpleNN等)
    X_train_flat, y_train_flat = X_train_scaled, y_train_scaled
    X_val_flat, y_val_flat = X_val_scaled, y_val_scaled
    X_test_flat, y_test_flat = X_test_scaled, y_test_scaled

    # 7. "Sequence" 数据 (用于EnhancedNN)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)
    
    # 8. 为保证flat数据与seq数据目标值y对齐，对flat数据进行截断
    # 这是因为 create_sequences 会让数据减少 sequence_length 的长度
    y_train_flat = y_train_flat[sequence_length:]
    y_val_flat = y_val_flat[sequence_length:]
    y_test_flat = y_test_flat[sequence_length:]

    # 对X_flat也进行相应截断
    X_train_flat = X_train_flat[sequence_length:]
    X_val_flat = X_val_flat[sequence_length:]
    X_test_flat = X_test_flat[sequence_length:]

    # 确保所有y是同步的
    assert np.array_equal(y_train_flat, y_train_seq)
    assert np.array_equal(y_val_flat, y_val_seq)
    assert np.array_equal(y_test_flat, y_test_seq)

    data = {
        'flat': {
            'train': (X_train_flat, y_train_flat),
            'val': (X_val_flat, y_val_flat),
            'test': (X_test_flat, y_test_flat)
        },
        'sequence': {
            'train': (X_train_seq, y_train_seq),
            'val': (X_val_seq, y_val_seq),
            'test': (X_test_seq, y_test_seq)
        },
        'scalers': (X_scaler, y_scaler),
        'feature_cols': feature_cols
    }
    return data

def train_and_predict_arima(train_series, test_series):
    """
    使用前向验证（Walk-forward validation）训练ARIMA并进行预测.
    这更耗时，但更准确，因为它在每个时间步都用最新的数据重新拟合。
    """
    history = [x for x in train_series]
    predictions = list()
    print("\n--- 训练 ARIMA 模型 (前向验证) ---")
    print("这可能会花费一些时间...")
    for t in range(len(test_series)):
        model = ARIMA(history, order=(5,1,0)) # p,d,q - 一个常用的基线阶数
        try:
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
        except Exception as e:
            # 如果模型拟合失败，使用最后一个有效预测值
            print(f"ARIMA在步骤 {t+1} 拟合失败: {e}。使用上一个预测值。")
            predictions.append(predictions[-1] if predictions else 0)

        obs = test_series.iloc[t]
        history.append(obs) # 将真实的观测值添加到历史数据中，用于下一次预测
        if (t+1) % 10 == 0:
            print(f"ARIMA 预测进度: {t+1}/{len(test_series)}")
    print("ARIMA 预测完成.")
    return np.array(predictions)

def train_prophet_model(train_df):
    """训练Prophet模型"""
    model = Prophet(
        yearly_seasonality=20,
        weekly_seasonality=10,
        daily_seasonality=False,
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=15,
        holidays_prior_scale=20,
        changepoint_range=0.9,
        seasonality_mode='additive'
    )
    
    model.add_seasonality(name='monthly', period=30.5, fourier_order=8)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=6)
    
    model.fit(train_df)
    return model

def train_ensemble_models(X_train, y_train):
    """训练集成模型"""
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgb_model.fit(X_train, y_train)

    # RandomForest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    return {'XGBoost': xgb_model, 'LightGBM': lgb_model, 'RandomForest': rf_model}

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden_states):
        # hidden_states: (batch_size, sequence_length, hidden_size)
        energy = torch.tanh(self.attn(hidden_states))
        # energy: (batch_size, sequence_length, hidden_size)
        energy = energy.transpose(1, 2)
        # energy: (batch_size, hidden_size, sequence_length)
        v = self.v.repeat(hidden_states.size(0), 1).unsqueeze(1)
        # v: (batch_size, 1, hidden_size)
        attention_scores = torch.bmm(v, energy).squeeze(1)
        # attention_scores: (batch_size, sequence_length)
        return F.softmax(attention_scores, dim=1)

class EnhancedNN(nn.Module):
    """LSTM + Attention 模型"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(EnhancedNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = Attention(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size * 2)
        
        attention_weights = self.attention(lstm_out)
        # attention_weights shape: (batch_size, sequence_length)
        
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        # context_vector shape: (batch_size, hidden_size * 2)
        
        x = F.relu(self.fc1(context_vector))
        x = self.dropout(x)
        out = self.fc2(x)
        return out.squeeze(1)

class SimpleNN(nn.Module):
    """一个简单的MLP模型"""
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze(-1)

def train_neural_network(model, X_train, y_train, X_val, y_val, epochs=500, batch_size=32, lr=0.001, patience=20, log_file_path=None, ablation_setup_name='N/A', model_name='N/A'):
    """
    通用神经网络训练函数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    val_data = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience//2, factor=0.5, verbose=True)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    log_header = not os.path.exists(log_file_path) if log_file_path else False

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}")

        scheduler.step(avg_val_loss)

        if log_file_path:
            with open(log_file_path, 'a') as f:
                if log_header:
                    f.write("timestamp,ablation_setup,model_name,epoch,train_loss,val_loss,learning_rate\n")
                    log_header = False # 只写一次
                
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{timestamp},{ablation_setup_name},{model_name},{epoch+1},{avg_train_loss},{avg_val_loss},{current_lr}\n")


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'best_new_data_{model_name}_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(f'best_new_data_{model_name}_model.pth'))
    return model

def plot_model_comparison(results, y_true, predictions):
    """绘制所有模型的预测结果与真实值的对比图"""
    plt.figure(figsize=(20, 12))
    plt.plot(y_true, label='真实值', color='black', linewidth=2, linestyle='--')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (model_name, data) in enumerate(predictions.items()):
        r2 = results[model_name]['R2']
        label = f"{model_name} (R²: {r2:.3f})"
        plt.plot(data, label=label, color=colors[i % len(colors)], alpha=0.8)
        
    plt.title('各模型预测结果对比', fontsize=20)
    plt.xlabel('时间步', fontsize=15)
    plt.ylabel('目标值', fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("new_data_model_predictions_comparison.png", dpi=300)
    print("\n模型预测对比图已保存为 'new_data_model_predictions_comparison.png'")

def calculate_permutation_importance(model, X_test, y_test, scaler):
    """
    计算神经网络模型的Permutation Feature Importance。
    这是一种模型无关的方法，通过打乱单个特征的顺序，观察模型性能的下降程度来评估特征的重要性。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().to(device)

    # 1. 计算基线性能
    with torch.no_grad():
        baseline_preds = model(X_test_tensor)
    
    # 逆缩放回原始尺度进行评估
    baseline_preds_rescaled = scaler.inverse_transform(baseline_preds.cpu().numpy().reshape(-1, 1)).flatten()
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    baseline_r2 = r2_score(y_test_rescaled, baseline_preds_rescaled)

    importances = []
    print("\n--- 计算 Permutation Feature Importance ---")
    
    num_features = X_test.shape[-1]
    
    for i in range(num_features):
        print(f"  正在计算特征 {i+1}/{num_features} 的重要性...")
        temp_X_test = X_test.copy()
        
        # 随机打乱第i个特征
        if len(temp_X_test.shape) == 3: # Sequence data
            np.random.shuffle(temp_X_test[:, :, i])
        else: # Flat data
            np.random.shuffle(temp_X_test[:, i])
            
        temp_X_test_tensor = torch.from_numpy(temp_X_test).float().to(device)
        
        with torch.no_grad():
            permuted_preds = model(temp_X_test_tensor)

        permuted_preds_rescaled = scaler.inverse_transform(permuted_preds.cpu().numpy().reshape(-1, 1)).flatten()
        permuted_r2 = r2_score(y_test_rescaled, permuted_preds_rescaled)
        
        # 重要性定义为性能下降的幅度
        importance = baseline_r2 - permuted_r2
        importances.append(importance)
        
    print("Permutation Feature Importance 计算完成.")
    return np.array(importances)

def plot_all_feature_importances(importance_dict, feature_names):
    """绘制所有模型（包括NN和集成模型）的特征重要性图"""
    num_models = len(importance_dict)
    if num_models == 0:
        print("没有可供可视化的特征重要性数据。")
        return
        
    fig, axes = plt.subplots(nrows=num_models, ncols=1, figsize=(15, 5 * num_models))
    if num_models == 1:
        axes = [axes] # make it iterable

    colors = ['#8c564b', '#9467bd', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (model_name, importances) in enumerate(importance_dict.items()):
        ax = axes[i]
        
        # 按重要性排序
        sorted_indices = np.argsort(importances)[::-1]
        
        # 只显示Top 15
        top_n = 15
        sorted_indices = sorted_indices[:top_n]
        
        sorted_importances = importances[sorted_indices]
        sorted_feature_names = [feature_names[j] for j in sorted_indices]
        
        bars = ax.barh(range(len(sorted_importances)), sorted_importances, align='center', color=colors[i % len(colors)])
        ax.set_yticks(range(len(sorted_importances)))
        ax.set_yticklabels(sorted_feature_names)
        ax.invert_yaxis()  # labels read top-to-bottom
        
        if 'Permutation' in model_name:
            ax.set_xlabel('重要性得分 (R²下降值)')
        else:
            ax.set_xlabel('重要性得分 (Importance Score)')

        ax.set_title(f'{model_name} - 特征重要性 (Top {top_n})', fontsize=16)

        # 在条形图上显示数值
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width * 1.01  # 在条形图右侧稍微偏移一点
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2e}', va='center')


    fig.suptitle('各模型特征重要性综合对比', fontsize=24, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig("new_data_all_models_feature_importance.png", dpi=300, bbox_inches='tight')
    print("\n所有模型的特征重要性图已保存为 'new_data_all_models_feature_importance.png'")


def evaluate_model(y_true, y_pred, model_name, nn_pred=None):
    """
    评估模型性能并返回指标字典。
    nn_pred: 对于神经网络，传入未逆缩放的预测值，用于计算MSE。
    """
    if y_true.ndim > 1: y_true = y_true.flatten()
    if y_pred.ndim > 1: y_pred = y_pred.flatten()

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 统一计算RMSE
    # 对于NN，我们需要用原始未缩放的预测值来计算MSE，因为y_pred是逆缩放过的
    if nn_pred is not None:
        if nn_pred.ndim > 1: nn_pred = nn_pred.flatten()
        # 假设y_true已经是逆缩放的了
        mse = mean_squared_error(y_true, nn_pred)
    else:
        mse = mean_squared_error(y_true, y_pred)
    
    rmse = np.sqrt(mse)
    
    print(f"--- {model_name} 评估结果 ---")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return {"R2": r2, "MAE": mae, "RMSE": rmse}

def save_log(message):
    with open("experiment_log.txt", "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

def main():
    # --- 数据加载与预处理 ---
    # 使用新的温度数据集
    try:
        df = pd.read_csv('min_daily_temps.csv')
    except FileNotFoundError:
        print("错误: 'min_daily_temps.csv' not found. 请确保已下载该文件。")
        return
    
    # 数据清洗和重命名
    # 将'Temp'列中的非数字值（例如'?'）转为NaN，然后移除这些行
    df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
    df.dropna(subset=['Temp'], inplace=True)
    
    # 为了复用代码，将列名重命名为脚本期望的名称
    df.rename(columns={'Date': '日期', 'Temp': '成交商品件数'}, inplace=True)
    df['日期'] = pd.to_datetime(df['日期'])

    # 定义模型和超参数
    SEQUENCE_LENGTH = 14
    TARGET_COL = '成交商品件数'

    # --- 1. 数据准备 ---
    print("\n--- 1. 准备数据 ---")
    data_dict = prepare_data(df, sequence_length=SEQUENCE_LENGTH)
    
    # 解包数据
    X_train_flat, y_train_flat = data_dict['flat']['train']
    X_val_flat, y_val_flat = data_dict['flat']['val']
    X_test_flat, y_test_flat = data_dict['flat']['test']
    
    X_train_seq, y_train_seq = data_dict['sequence']['train']
    X_val_seq, y_val_seq = data_dict['sequence']['val']
    X_test_seq, y_test_seq = data_dict['sequence']['test']
    
    X_scaler, y_scaler = data_dict['scalers']
    feature_cols = data_dict['feature_cols']

    results = {}
    predictions = {}
    feature_importances = {}
    
    # --- 2. 传统时序模型 ---
    print("\n--- 2. 训练传统时序模型 ---")
    # 准备Prophet的数据
    df_prophet = df[['日期', TARGET_COL]].rename(columns={'日期': 'ds', '成交商品件数': 'y'})
    train_size = int(len(df_prophet) * 0.7)
    train_df_prophet = df_prophet.iloc[:train_size]
    test_df_prophet = df_prophet.iloc[train_size:]
    
    # Prophet
    prophet_model = train_prophet_model(train_df_prophet)
    future = prophet_model.make_future_dataframe(periods=len(test_df_prophet), freq='D')
    prophet_forecast = prophet_model.predict(future)
    prophet_preds_aligned = prophet_forecast['yhat'].values[-len(y_test_flat):]
    
    # 准备ARIMA的数据
    arima_series = df[TARGET_COL]
    train_series_arima = arima_series[:train_size]
    test_series_arima = arima_series[train_size:]
    test_series_arima_aligned = test_series_arima[-len(y_test_flat):] # 对齐

    # ARIMA
    arima_preds = train_and_predict_arima(train_series_arima, test_series_arima)
    arima_preds_aligned = arima_preds[-len(y_test_flat):]

    # 评估
    y_test_rescaled = y_scaler.inverse_transform(y_test_flat.reshape(-1, 1)).flatten()
    
    results['Prophet'] = evaluate_model(y_test_rescaled, prophet_preds_aligned, "Prophet")
    predictions['Prophet'] = prophet_preds_aligned
    
    results['ARIMA'] = evaluate_model(y_test_rescaled, arima_preds_aligned, "ARIMA")
    predictions['ARIMA'] = arima_preds_aligned

    # --- 3. 集成学习模型 ---
    print("\n--- 3. 训练集成学习模型 ---")
    ensemble_models = train_ensemble_models(X_train_flat, y_train_flat)
    
    for name, model in ensemble_models.items():
        preds_scaled = model.predict(X_test_flat)
        preds_rescaled = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        
        results[name] = evaluate_model(y_test_rescaled, preds_rescaled, name)
        predictions[name] = preds_rescaled
        feature_importances[name] = model.feature_importances_

    # --- 4. 神经网络模型 ---
    print("\n--- 4. 训练神经网络模型 ---")
    
    # SimpleNN
    simple_nn = SimpleNN(X_train_flat.shape[1])
    simple_nn = train_neural_network(simple_nn, X_train_flat, y_train_flat, X_val_flat, y_val_flat, model_name='SimpleNN', log_file_path='main_training_log.csv')
    
    # EnhancedNN
    enhanced_nn = EnhancedNN(X_train_seq.shape[2])
    enhanced_nn = train_neural_network(enhanced_nn, X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_name='EnhancedNN', log_file_path='main_training_log.csv')
    
    # 评估NN模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SimpleNN 评估
    simple_nn_preds_scaled = simple_nn(torch.from_numpy(X_test_flat).float().to(device)).cpu().detach().numpy()
    simple_nn_preds_rescaled = y_scaler.inverse_transform(simple_nn_preds_scaled.reshape(-1, 1)).flatten()
    results['SimpleNN'] = evaluate_model(y_test_rescaled, simple_nn_preds_rescaled, "SimpleNN", nn_pred=simple_nn_preds_scaled)
    predictions['SimpleNN'] = simple_nn_preds_rescaled
    
    # EnhancedNN 评估
    enhanced_nn_preds_scaled = enhanced_nn(torch.from_numpy(X_test_seq).float().to(device)).cpu().detach().numpy()
    enhanced_nn_preds_rescaled = y_scaler.inverse_transform(enhanced_nn_preds_scaled.reshape(-1, 1)).flatten()
    results['EnhancedNN'] = evaluate_model(y_test_rescaled, enhanced_nn_preds_rescaled, "EnhancedNN", nn_pred=enhanced_nn_preds_scaled)
    predictions['EnhancedNN'] = enhanced_nn_preds_rescaled

    # --- 5. 计算NN模型的特征重要性 ---
    print("\n--- 5. 计算神经网络模型的特征重要性 ---")
    
    # SimpleNN
    importance_simple = calculate_permutation_importance(simple_nn, X_test_flat, y_test_flat, y_scaler)
    feature_importances['SimpleNN (Permutation)'] = importance_simple
    
    # EnhancedNN
    importance_enhanced = calculate_permutation_importance(enhanced_nn, X_test_seq, y_test_seq, y_scaler)
    feature_importances['EnhancedNN (Permutation)'] = importance_enhanced
    
    # --- 6. 可视化与保存 ---
    print("\n--- 6. 生成可视化图表 ---")
    plot_model_comparison(results, y_test_rescaled, predictions)
    plot_all_feature_importances(feature_importances, feature_cols)

    # 打印最终R²分数对比
    print("\n--- 所有模型最终 R² 分数对比 ---")
    for name, result in sorted(results.items(), key=lambda item: item[1]['R2'], reverse=True):
        print(f"{name}: {result['R2']:.4f}")

if __name__ == '__main__':
    main() 