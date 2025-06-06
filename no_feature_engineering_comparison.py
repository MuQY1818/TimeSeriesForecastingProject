import pandas as pd
import numpy as np
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
    """(实验) 创建一个简化的特征集，仅包含滞后值，以测试模型在没有复杂特征工程下的表现。"""
    df_copy = df.copy()
    df_copy['日期'] = pd.to_datetime(df_copy['日期'])
    
    # 仅创建滞后特征
    for i in range(1, 15):
        df_copy[f'lag_{i}'] = df_copy[target_col].shift(i)
    
    # 处理因滞后操作产生的缺失值
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
    best_model_state = None
    no_improve_count = 0
    history = []
    
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
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        history.append((epoch, avg_train_loss, val_loss))
    
    # 加载最佳模型权重并返回
    if best_model_state:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: Early stopping was not triggered. Returning model from last epoch.")

    return model, history

def plot_model_comparison(results, y_true, predictions):
    """
    绘制模型比较图，包括预测值、真实值、误差和性能指标。
    """
    model_names = list(results.keys())
    num_models = len(model_names)
    
    # 获取颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, num_models))
    model_colors = {name: color for name, color in zip(model_names, colors)}

    # 创建一个 (3, 1) 的子图布局
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 20), 
                                       gridspec_kw={'height_ratios': [3, 1, 1.5]})
    
    # --- 子图1: 实际值 vs 预测值 ---
    ax1.plot(y_true, label='实际值', color='black', linewidth=2, linestyle='--')
    for model_name, pred in predictions.items():
        ax1.plot(pred, label=f'{model_name} 预测', color=model_colors[model_name], alpha=0.8)
    ax1.set_title('模型预测对比', fontsize=16)
    ax1.set_ylabel('成交商品件数', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # --- 子图2: 预测误差 ---
    for model_name, pred in predictions.items():
        error = y_true - pred
        ax2.plot(error, label=f'{model_name} 误差', color=model_colors[model_name], alpha=0.7)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_title('预测误差 (真实值 - 预测值)', fontsize=16)
    ax2.set_ylabel('误差', fontsize=12)
    ax2.legend()
    ax2.grid(True)

    # --- 子图3: 模型性能指标 ---
    ax3.axis('off') # 关闭坐标轴
    
    # 创建表格数据
    table_data = []
    headers = ["模型", "RMSE", "MAE", "MAPE (%)", "sMAPE (%)", "$R^2$", "改进率 (%)"]
    
    # 获取基准模型的RMSE
    baseline_rmse = results.get('SimpleNN', {}).get('RMSE', None)

    for name, metrics in results.items():
        # 创建一个安全的格式化函数，避免对非数字值进行格式化
        def safe_format(value, format_str):
            return format(value, format_str) if isinstance(value, (int, float)) else 'N/A'

        row = [
            name,
            safe_format(metrics.get('RMSE'), '.2f'),
            safe_format(metrics.get('MAE'), '.2f'),
            safe_format(metrics.get('MAPE'), '.2f'),
            safe_format(metrics.get('sMAPE'), '.2f'),
            safe_format(metrics.get('R2'), '.4f'),
        ]
        
        # 计算相对改进率
        if baseline_rmse and name != 'SimpleNN' and 'RMSE' in metrics and isinstance(metrics['RMSE'], (int, float)):
            improvement = ((baseline_rmse - metrics['RMSE']) / baseline_rmse) * 100
            row.append(f"{improvement:.2f}")
        elif name == 'SimpleNN':
             row.append("基准")
        else:
            row.append("N/A")
            
        table_data.append(row)
        
    # 在子图3中添加表格
    table = ax3.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2) # 拉伸表格高度
    ax3.set_title('模型性能评估指标', fontsize=16, y=0.85)

    # 全局标题和布局调整
    fig.suptitle('无特征工程: 各模型性能综合对比', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图像
    save_path = 'models/no_fe_model_comparison.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True) # 确保目录存在
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n对比图已保存至 {save_path}")
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
    
    # 计算相对于基准模型(SimpleNN)的改进率
    improvement_rate = 'N/A'
    if nn_pred is not None:
        nn_rmse = np.sqrt(mean_squared_error(y_true, nn_pred))
        improvement_rate = f"{((nn_rmse - rmse) / nn_rmse) * 100:.2f}%"

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "sMAPE": smape,
        "R2": r2,
        "Improvement": improvement_rate
    }
    
    print(f"\n{model_name} 模型评估指标：")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}" + ("%" if "MAPE" in key else ""))
        else:
            print(f"{key}: {value}")
            
    return metrics

def main():
    """主函数"""
    # 确保保存模型的目录存在
    os.makedirs('models', exist_ok=True)

    print("--- 开始无特征工程对比实验 ---")
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
    print("\n--- 训练传统模型 (无特征工程) ---")
    xgb_model, lgb_model, rf_model = train_ensemble_models(X_train_flat, y_train_flat)
    
    print("\n--- 训练神经网络模型 (无特征工程) ---\n")
    input_size_simple = data['flat']['train'][0].shape[1]
    input_size_enhanced = data['seq']['train'][0].shape[2]

    # 初始化两个神经网络模型
    simple_nn = SimpleNN(input_size=input_size_simple)
    enhanced_nn = EnhancedNN(input_size=input_size_enhanced, hidden_size=64, num_layers=2, dropout=0.3)
    
    # 训练SimpleNN
    print("训练 SimpleNN...")
    simple_nn, _ = train_neural_network(
        simple_nn, 
        data['flat']['train'][0], data['flat']['train'][1],
        data['flat']['val'][0], data['flat']['val'][1],
        epochs=1000, batch_size=32, lr=0.001, patience=50
    )
    
    # 训练EnhancedNN
    print("\n训练 EnhancedNN...")
    enhanced_nn, _ = train_neural_network(
        enhanced_nn,
        data['seq']['train'][0], data['seq']['train'][1],
        data['seq']['val'][0], data['seq']['val'][1],
        epochs=1000, batch_size=32, lr=0.001, patience=50
    )
    
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
    joblib.dump(xgb_model, 'models/no_fe_xgb_model.joblib')
    joblib.dump(lgb_model, 'models/no_fe_lgb_model.joblib')
    joblib.dump(rf_model, 'models/no_fe_rf_model.joblib')
    joblib.dump(X_scaler, 'models/no_fe_X_scaler.joblib')
    joblib.dump(y_scaler, 'models/no_fe_y_scaler.joblib')
    torch.save(simple_nn.state_dict(), 'models/no_fe_simple_nn_model.pth')
    torch.save(enhanced_nn.state_dict(), 'models/no_fe_enhanced_nn_model.pth')
    
    # 保存预测结果
    results_df = pd.DataFrame(predictions)
    results_df['实际值'] = y_test_orig
    results_df.to_csv('models/no_fe_model_predictions.csv', index=False)
    
    print("\n所有(无特征工程)模型和结果已保存到models目录")
    print("--- 无特征工程对比实验结束 ---")

if __name__ == "__main__":
    main() 