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
    
    # 移动平均特征
    for window in [7, 14, 30]:
        df_copy[f'rolling_mean_{window}'] = df_copy[target_col].rolling(window=window, min_periods=1).mean()
        df_copy[f'rolling_std_{window}'] = df_copy[target_col].rolling(window=window, min_periods=1).std()
        df_copy[f'ma_ratio_{window}'] = df_copy[target_col] / df_copy[f'rolling_mean_{window}']
    
    # 差分特征
    df_copy['diff_1'] = df_copy[target_col].diff()
    df_copy['diff_7'] = df_copy[target_col].diff(7)
    
    # 趋势特征
    df_copy['trend'] = df_copy[target_col].rolling(window=7, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    
    # 处理缺失值
    df_copy = df_copy.fillna(method='bfill').fillna(method='ffill')
    
    return df_copy

def create_sequences(X, y, sequence_length=14):
    """为序列模型创建时间序列样本"""
    xs, ys = [], []
    for i in range(len(X) - sequence_length):
        xs.append(X[i:(i + sequence_length)])
        ys.append(y[i + sequence_length])
    return np.array(xs), np.array(ys)

def prepare_data(df, target_col='成交商品件数', train_ratio=0.8, val_ratio=0.1, sequence_length=14):
    """准备模型数据"""
    # 创建高级特征
    df_features = create_advanced_features(df, target_col)
    
    # 准备其他模型的特征
    feature_cols = [col for col in df_features.columns 
                   if col not in ['日期', target_col]]
    X = df_features[feature_cols].values
    y = df_features[target_col].values
    
    # 数据分割点
    total_size = len(X)
    train_end = int(total_size * train_ratio)
    val_end = train_end + int(total_size * val_ratio)
    
    # 数据缩放 (仅用训练集拟合)
    X_scaler = RobustScaler().fit(X[:train_end])
    y_scaler = RobustScaler().fit(y[:train_end].reshape(-1, 1))
    
    # 缩放整个数据集
    X_scaled = X_scaler.transform(X)
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).flatten()
    
    # 分割数据集
    X_train_scaled, y_train_scaled = X_scaled[:train_end], y_scaled[:train_end]
    X_val_scaled, y_val_scaled = X_scaled[train_end:val_end], y_scaled[train_end:val_end]
    X_test_scaled, y_test_scaled = X_scaled[val_end:], y_scaled[val_end:]
    
    # --- 为不同类型的模型创建对齐的数据 ---
    # 1. "Flat" 数据 (用于XGBoost, RF, SimpleNN等)
    # 我们从 `sequence_length` 开始索引，以确保目标值 `y` 与序列模型的输出对齐
    X_train_flat = X_train_scaled[sequence_length:]
    y_train_flat = y_train_scaled[sequence_length:]
    X_val_flat, y_val_flat = X_val_scaled[sequence_length:], y_val_scaled[sequence_length:]
    X_test_flat, y_test_flat = X_test_scaled[sequence_length:], y_test_scaled[sequence_length:]
    
    # 2. "Sequence" 数据 (用于EnhancedNN)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)

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
        'seq': {
            'train': (X_train_seq, y_train_seq),
            'val': (X_val_seq, y_val_seq),
            'test': (X_test_seq, y_test_seq)
        },
        'scalers': (X_scaler, y_scaler),
        'feature_cols': feature_cols
    }
    return data

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
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # RandomForest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # 训练模型
    print("训练XGBoost...")
    xgb_model.fit(X_train, y_train.ravel())
    
    print("训练LightGBM...")
    lgb_model.fit(X_train, y_train.ravel())
    
    print("训练随机森林...")
    rf_model.fit(X_train, y_train.ravel())
    
    return xgb_model, lgb_model, rf_model

class EnhancedNN(nn.Module):
    """增强型神经网络，结合LSTM和Attention机制"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(EnhancedNN, self).__init__()
        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout
        )
        # Attention层
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        # 输出层
        self.output_fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size * 2)
        
        # Attention机制
        # 1. 计算每个时间步的Attention分数
        attention_scores = self.attention_fc(lstm_out)
        # attention_scores shape: (batch_size, sequence_length, 1)
        
        # 2. 将分数转换为概率分布 (weights)
        attention_weights = F.softmax(attention_scores, dim=1)
        # attention_weights shape: (batch_size, sequence_length, 1)

        # 3. 计算上下文向量 (context vector)
        # (batch_size, 1, sequence_length) * (batch_size, sequence_length, hidden_size * 2)
        # -> (batch_size, 1, hidden_size * 2)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), lstm_out).squeeze(1)
        # context_vector shape: (batch_size, hidden_size * 2)
        
        # 通过输出层进行最终预测
        output = self.dropout(context_vector)
        output = self.output_fc(output)
        # output shape: (batch_size, 1)
        return output

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_neural_network(model, X_train, y_train, X_val, y_val, epochs=500, batch_size=32, lr=0.001, patience=20):
    """训练神经网络模型 (通用版)"""
    # 准备数据
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型、损失函数和优化器
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # 早停设置
    best_val_loss = float('inf')
    best_model = None
    no_improve_count = 0
    
    # 训练模型
    model.train()
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.view(-1, 1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor.view(-1, 1))
            
        # 学习率调整
        scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # 加载最佳模型
    model.load_state_dict(best_model)
    return model

def plot_model_comparison(results, y_true, predictions):
    """
    绘制模型比较图，现在包含预测对比和误差分析两个子图。
    """
    model_names = list(results.keys())
    num_models = len(model_names)
    
    # 获取颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, num_models))
    model_colors = {name: color for name, color in zip(model_names, colors)}

    # 创建一个 (2, 1) 的子图布局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # --- 子图1: 实际值 vs 预测值 ---
    ax1.plot(y_true, label='实际值', color='black', linewidth=2, linestyle='--')
    for model_name, pred in predictions.items():
        ax1.plot(pred, label=f'{model_name} 预测', color=model_colors[model_name], alpha=0.8)
    ax1.set_title('模型预测对比 (有特征工程)', fontsize=16)
    ax1.set_ylabel('成交商品件数', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # --- 子图2: 预测误差 ---
    for model_name, pred in predictions.items():
        error = y_true - pred
        ax2.plot(error, label=f'{model_name} 误差', color=model_colors[model_name], alpha=0.7)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_title('预测误差 (真实值 - 预测值)', fontsize=16)
    ax2.set_xlabel('测试集样本索引', fontsize=12)
    ax2.set_ylabel('误差', fontsize=12)
    ax2.grid(True)

    # 调整整体布局并保存
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle('特征工程下各模型性能综合对比', fontsize=20, y=0.99)
    
    save_path = 'models/model_comparison.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n对比图已保存至 {save_path}")
    plt.close(fig)

def calculate_permutation_importance(model, X_test, y_test, scaler):
    """
    计算神经网络模型的置换特征重要性。
    返回一个包含各特征重要性分数的Numpy数组。
    """
    print(f"\n--- 正在计算 {model.__class__.__name__} 的置换特征重要性... ---")
    
    model.eval()

    # 计算基准性能
    is_sequence = len(X_test.shape) == 3
    X_test_tensor = torch.FloatTensor(X_test)
    with torch.no_grad():
        base_preds_scaled = model(X_test_tensor).numpy()
    base_preds = scaler.inverse_transform(base_preds_scaled)
    base_mse = mean_squared_error(y_test, base_preds)
    
    importances = []
    # 特征数量在最后一个维度
    num_features = X_test.shape[-1]

    for i in range(num_features):
        X_permuted = X_test.copy()
        
        # 根据数据是扁平的还是序列化的，用不同方式打乱第i个特征
        if is_sequence:
            # 对于序列数据 (samples, seq_len, features)
            # 我们打乱每个样本和时间步的第i个特征值
            original_shape = X_permuted[:, :, i].shape
            X_permuted[:, :, i] = np.random.permutation(X_permuted[:, :, i].flatten()).reshape(original_shape)
        else:
            # 对于扁平数据 (samples, features)
            np.random.shuffle(X_permuted[:, i])
        
        X_permuted_tensor = torch.FloatTensor(X_permuted)
        with torch.no_grad():
            permuted_preds_scaled = model(X_permuted_tensor).numpy()
        
        permuted_preds = scaler.inverse_transform(permuted_preds_scaled)
        permuted_mse = mean_squared_error(y_test, permuted_preds)
        
        # 重要性分数为MSE的增加值
        importance = permuted_mse - base_mse
        importances.append(importance)
        
    print(f"--- {model.__class__.__name__} 重要性计算完成 ---")
    return np.array(importances)

def plot_all_feature_importances(importance_dict, feature_names):
    """
    绘制所有模型的特征重要性对比图，采用 3x2 网格布局。
    """
    model_names = list(importance_dict.keys())
    
    # 创建一个 3x2 的网格布局
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    axes = axes.flatten() # 将 3x2 的二维数组展平为一维

    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    model_colors = {name: color for name, color in zip(model_names, colors)}

    for i, name in enumerate(model_names):
        ax = axes[i]
        importances = importance_dict[name]
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(15)

        ax.barh(feature_importance_df['feature'], feature_importance_df['importance'], color=model_colors[name], alpha=0.8)
        ax.set_title(f'{name} - 特征重要性 (Top 15)', fontsize=16)
        ax.invert_yaxis()
        ax.set_xlabel('重要性得分 (Importance Score)', fontsize=12)

    # 隐藏最后一个未使用的子图
    if len(model_names) < len(axes):
        for i in range(len(model_names), len(axes)):
            fig.delaxes(axes[i])

    fig.suptitle('各模型特征重要性综合对比', fontsize=24, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    save_path = 'models/all_models_feature_importance.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n综合特征重要性图已保存至 {save_path}")
    plt.close(fig)

def evaluate_model(y_true, y_pred, model_name, nn_pred=None):
    """评估模型性能"""
    # 替换无穷大值
    y_true = np.where(np.isinf(y_true), 0, y_true)
    y_pred = np.where(np.isinf(y_pred), 0, y_pred)
    
    # 替换NaN值
    y_true = np.where(np.isnan(y_true), 0, y_true)
    y_pred = np.where(np.isnan(y_pred), 0, y_pred)
    
    # 计算sMAPE
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # 处理分母为零的情况
    safe_denominator = np.where(denominator == 0, 1, denominator)
    smape = np.mean(numerator / safe_denominator) * 100

    # 计算MAPE
    mape_denominator = np.abs(y_true)
    safe_mape_denominator = np.where(mape_denominator == 0, 1, mape_denominator)
    mape = np.mean(np.abs((y_true - y_pred) / safe_mape_denominator)) * 100
    
    # 计算其他指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 如果是集成模型，计算相对于神经网络的改进
    if model_name != 'NeuralNetwork' and nn_pred is not None:
        nn_error = np.abs(y_true - nn_pred)
        ensemble_error = np.abs(y_true - y_pred)
        # 避免除以零
        safe_nn_error = np.where(nn_error == 0, 1, nn_error)
        improvement = (nn_error - ensemble_error) / safe_nn_error * 100
        improvement_rate = np.mean(improvement)
        print(f"相对于神经网络的改进率: {improvement_rate:.2f}%")
    
    print(f"\n{model_name} 模型评估指标：")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
    print(f"对称平均绝对百分比误差 (sMAPE): {smape:.2f}%")
    print(f"R² 分数: {r2:.4f}")
    
    return {
        'model': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'smape': smape,
        'r2': r2
    }

def main():
    # 加载数据
    df = pd.read_csv('total_cleaned.csv')
    SEQUENCE_LENGTH = 14
    
    # 准备数据
    data = prepare_data(df, sequence_length=SEQUENCE_LENGTH)
    
    # 解包数据
    X_train_flat, y_train_flat = data['flat']['train']
    X_val_flat, y_val_flat = data['flat']['val']
    X_test_flat, y_test_flat = data['flat']['test']
    
    X_train_seq, y_train_seq = data['seq']['train']
    X_val_seq, y_val_seq = data['seq']['val']
    X_test_seq, y_test_seq = data['seq']['test']

    X_scaler, y_scaler = data['scalers']

    print("数据集大小:")
    print(f"训练集 (Flat/Seq): {len(X_train_flat)} / {len(X_train_seq)}")
    print(f"验证集 (Flat/Seq): {len(X_val_flat)} / {len(X_val_seq)}")
    print(f"测试集 (Flat/Seq): {len(X_test_flat)} / {len(X_test_seq)}")
    
    # --- 训练模型 ---
    print("\n--- 训练传统模型 ---")
    xgb_model, lgb_model, rf_model = train_ensemble_models(X_train_flat, y_train_flat)
    
    print("\n--- 训练神经网络模型 ---")
    # 1. 训练简单神经网络 (SimpleNN)
    print("\n训练 SimpleNN...")
    simple_nn_model = SimpleNN(input_size=X_train_flat.shape[1])
    simple_nn = train_neural_network(simple_nn_model, X_train_flat, y_train_flat, X_val_flat, y_val_flat)

    # 2. 训练增强型神经网络 (EnhancedNN)
    print("\n训练 EnhancedNN...")
    enhanced_nn_model = EnhancedNN(input_size=X_train_seq.shape[2])
    enhanced_nn = train_neural_network(enhanced_nn_model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, lr=0.0001)
    
    # --- 评估模型 ---
    # 获取用于评估的真实y值 (并逆缩放)
    y_test_orig = y_scaler.inverse_transform(y_test_flat.reshape(-1, 1)).flatten()
    
    results = {}
    predictions = {}
    
    all_models = {
        'XGBoost': xgb_model,
        'LightGBM': lgb_model,
        'RandomForest': rf_model,
        'SimpleNN': simple_nn,
        'EnhancedNN': enhanced_nn
    }
    
    # 提取EnhancedNN的预测，用于对比
    enhanced_nn.eval()
    with torch.no_grad():
        enhanced_nn_preds_scaled = enhanced_nn(torch.FloatTensor(X_test_seq)).numpy()
    enhanced_nn_preds_orig = y_scaler.inverse_transform(enhanced_nn_preds_scaled).flatten()

    for name, model in all_models.items():
        if name in ['XGBoost', 'LightGBM', 'RandomForest']:
            pred_scaled = model.predict(X_test_flat).reshape(-1,1)
        elif name == 'SimpleNN':
            model.eval()
            with torch.no_grad():
                pred_scaled = model(torch.FloatTensor(X_test_flat)).numpy()
        else: # EnhancedNN
            pred_scaled = enhanced_nn_preds_scaled

        pred_orig = y_scaler.inverse_transform(pred_scaled).flatten()
        
        # 使用EnhancedNN作为基准进行比较
        results[name] = evaluate_model(y_test_orig, pred_orig, name, nn_pred=enhanced_nn_preds_orig)
        predictions[name] = pred_orig
    
    # 绘制对比结果
    plot_model_comparison(results, y_test_orig, predictions)
    
    # 保存模型和结果
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_model, 'models/xgb_model.joblib')
    joblib.dump(lgb_model, 'models/lgb_model.joblib')
    joblib.dump(rf_model, 'models/rf_model.joblib')
    joblib.dump(X_scaler, 'models/X_scaler.joblib')
    joblib.dump(y_scaler, 'models/y_scaler.joblib')
    torch.save(simple_nn.state_dict(), 'models/simple_nn_model.pth')
    torch.save(enhanced_nn.state_dict(), 'models/enhanced_nn_model.pth')
    
    # 保存预测结果
    results_df = pd.DataFrame(predictions)
    results_df['实际值'] = y_test_orig
    results_df.to_csv('models/model_predictions.csv', index=False)
    
    # --- 进行可解释性分析 ---
    print("\n--- 开始进行模型可解释性分析 ---")
    
    # 1. 计算所有模型的特征重要性
    importances_xgb = xgb_model.feature_importances_
    importances_lgbm = lgb_model.feature_importances_
    importances_rf = rf_model.feature_importances_

    importances_simple_nn = calculate_permutation_importance(
        simple_nn, 
        data['flat']['test'][0],
        y_test_orig,
        data['scalers'][1] # y_scaler
    )
    
    importances_enhanced_nn = calculate_permutation_importance(
        enhanced_nn,
        data['seq']['test'][0],
        y_test_orig,
        data['scalers'][1] # y_scaler
    )

    # 2. 汇总并绘图
    all_importances = {
        'RandomForest': importances_rf,
        'XGBoost': importances_xgb,
        'LightGBM': importances_lgbm,
        'SimpleNN (Permutation)': importances_simple_nn,
        'EnhancedNN (Permutation)': importances_enhanced_nn
    }
    
    plot_all_feature_importances(all_importances, data['feature_cols'])

    print("\n所有模型和结果已保存到models目录")

if __name__ == "__main__":
    main()