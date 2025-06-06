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
    """绘制模型对比结果"""
    # 重新设计图表布局为2x2网格，以突出核心对比
    fig = plt.figure(figsize=(18, 12))
    gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

    ax1 = fig.add_subplot(gs[0, :])  # [0, :] 表示占据第一行的所有列
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 0])

    # --- 子图1 (ax1): 所有模型预测总览 ---
    colors = {
        'XGBoost': '#1f77b4', 'LightGBM': '#ff7f0e', 'RandomForest': '#2ca02c',
        'SimpleNN': '#9467bd', 'EnhancedNN': '#d62728',
    }
    
    # 绘制所有模型的预测线，创建视觉层次
    for model_name, pred in predictions.items():
        if model_name == 'SimpleNN':
            # 最重要的对比模型，使用独特且显眼的虚线
            style = {'linestyle': '--', 'linewidth': 2.2, 'alpha': 1.0, 'zorder': 9}
        elif model_name == 'RandomForest':
            # 表现也很好，使用另一种独特的划线
            style = {'linestyle': '-.', 'linewidth': 2.0, 'alpha': 0.8, 'zorder': 8}
        else: # XGBoost, LightGBM, EnhancedNN 作为背景
            style = {'linestyle': ':', 'linewidth': 1.5, 'alpha': 0.5, 'zorder': 5}
            
        ax1.plot(pred, label=model_name, color=colors.get(model_name, '#7f7f7f'), **style)

    # 绘制实际值，使其最突出
    ax1.plot(y_true, label='实际值', color='black', linewidth=2.5, marker='o', 
             markersize=3, markerfacecolor='white', markeredgecolor='black', zorder=10)
    
    ax1.set_title('各模型预测结果总览', fontsize=16, pad=15)
    ax1.set_xlabel('测试集样本索引', fontsize=12)
    ax1.set_ylabel('销量', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 子图2 (ax2): 误差指标对比 (使用对数尺度) ---
    metrics = ['rmse', 'mae', 'smape', 'r2']
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names) # 动态调整柱子宽度
    
    for i, name in enumerate(model_names):
        # 对R²分数进行特殊处理，因为它可能为负，不适合对数尺度
        metric_values = [results[name][m] if m != 'r2' else max(results[name][m], 0.001) for m in metrics]
        pos = x - (width * len(model_names) / 2) + i * width
        bars = ax2.bar(pos, metric_values, width, label=name, color=colors.get(name, '#7f7f7f'))
        # 在普通坐标中标注文本
        for bar, metric_name in zip(bars, metrics):
            true_value = results[name][metric_name]
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{true_value:.2f}',
                     ha='center', va='bottom', fontsize=8, rotation=45)

    ax2.set_yscale('log') # 使用对数尺度
    # 使用Matplotlib的MathText来渲染R²，避免乱码
    ax2.set_title('各模型误差指标对比 (对数尺度, $R^2$除外)', fontsize=16, pad=15)
    ax2.set_ylabel('误差值 (log scale)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in metrics])
    ax2.legend(title='模型', fontsize=10)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')

    # --- 子图3 (ax3): 核心焦点对比 (SimpleNN vs EnhancedNN) ---
    ax3.plot(y_true, label='实际值', color='black', linewidth=2.5, zorder=10)
    ax3.plot(predictions['SimpleNN'], label='SimpleNN 预测', color=colors['SimpleNN'], linestyle='--', linewidth=2)
    ax3.plot(predictions['EnhancedNN'], label='EnhancedNN 预测', color=colors['EnhancedNN'], linestyle=':', linewidth=2, alpha=0.8)
    
    ax3.set_title('核心对比: SimpleNN vs. EnhancedNN', fontsize=16, pad=15)
    ax3.set_xlabel('测试集样本索引', fontsize=12)
    ax3.set_ylabel('销量', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # 调整整体布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle('模型性能综合评估', fontsize=20)
    
    # 保存图片
    plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

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
    
    print("\n所有模型和结果已保存到models目录")

if __name__ == "__main__":
    main()