import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datasetsforecast.m5 import M5
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import os
import torch.nn.functional as F
from torch.optim import lr_scheduler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. DATASET LOADER
# ==============================================================================
def load_dataset(name: str):
    """
    MODIFIED: Now loads one of four datasets:
    - 'sales_volume': Original sales data.
    - 'min_daily_temps': Daily minimum temperatures.
    - 'etth1': Hourly electricity transformer temperature data.
    - 'wiki_traffic': Daily traffic for a specific Wikipedia page.
    """
    print(f"--- 1. 正在加载数据集: {name} ---")
    
    if name == 'sales_volume':
        try:
            df = pd.read_csv('data/total_cleaned.csv', encoding='gbk')
            df.rename(columns={'日期': 'ds', '成交商品件数': 'y'}, inplace=True)
            df['unique_id'] = 'Sales'
        except FileNotFoundError:
            print("错误: 'data/total_cleaned.csv' 未找到。")
            return None
    
    elif name == 'min_daily_temps':
        try:
            df = pd.read_csv('data/daily-minimum-temperatures.csv', parse_dates=['Date'])
            df.rename(columns={'Date': 'ds', 'Temp': 'y'}, inplace=True)
            df['unique_id'] = 'T1'
        except FileNotFoundError:
            print("错误: 'data/daily-minimum-temperatures.csv' 未找到。")
            return None

    elif name in ['etth1', 'wiki_traffic']:
        try:
            from datasetsforecast.m4 import M4
            from datasetsforecast.long_horizon import LongHorizon

            if name == 'etth1':
                # ETTh1 is hourly data
                Y_df, _, _ = LongHorizon.load(directory='.', group='ETTh1')
                # Let's select one of the transformer series, e.g., 'OT'
                df = Y_df[Y_df['unique_id'] == 'OT']
            
            elif name == 'wiki_traffic':
                # For demonstration, let's take a single interesting time series from M4 Daily
                # This requires downloading the M4 daily dataset
                Y_df, _, _ = M4.load(directory='.', group='Daily')
                # Let's pick a series with clear patterns, e.g., 'D414'
                df = Y_df[Y_df['unique_id'] == 'D414']

        except ImportError:
            print("错误: 'datasetsforecast' 未安装。请运行 'pip install datasetsforecast'")
            return None
        except Exception as e:
            print(f"从 datasetsforecast 加载数据时出错: {e}")
            return None

    else:
        print(f"错误: 未知的数据集名称 '{name}'。")
        return None
        
    df['ds'] = pd.to_datetime(df['ds'])
    df = df[['unique_id', 'ds', 'y']].sort_values('ds').reset_index(drop=True)
    df['y'] = pd.to_numeric(df['y'], errors='coerce') # Ensure 'y' is numeric
    df.dropna(subset=['y'], inplace=True)

    print(f"✅ {name} 加载成功. Shape: {df.shape}. Time range: {df['ds'].min()} to {df['ds'].max()}")
    return df

# ==============================================================================
# 2. FEATURE ENGINEERING & DATA PREP (New Legacy-Compliant Function)
# ==============================================================================
def create_advanced_features(df, target_col='y'):
    """
    MODIFIED: Upgraded to the more sophisticated feature engineering from the original
    'model_comparison.py' script. This includes dense lags, cyclical features,
    and trend/difference features to improve model performance.
    """
    print("  正在使用增强版特征工程...")
    df_copy = df.copy()
    
    # 确保 'ds' 是日期时间类型
    df_copy['ds'] = pd.to_datetime(df_copy['ds'])
    
    # 基础时间特征
    df_copy['dayofweek'] = df_copy['ds'].dt.dayofweek
    df_copy['is_weekend'] = df_copy['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df_copy['month'] = df_copy['ds'].dt.month
    df_copy['day'] = df_copy['ds'].dt.day
    
    # 周期性特征
    df_copy['sin_day'] = np.sin(2 * np.pi * df_copy['ds'].dt.day / 31)
    df_copy['cos_day'] = np.cos(2 * np.pi * df_copy['ds'].dt.day / 31)
    df_copy['sin_month'] = np.sin(2 * np.pi * df_copy['ds'].dt.month / 12)
    df_copy['cos_month'] = np.cos(2 * np.pi * df_copy['ds'].dt.month / 12)
    
    # 密集的滞后特征
    print("    - 创建1-14天的密集滞后特征")
    for i in range(1, 15):
        df_copy[f'lag_{i}'] = df_copy[target_col].shift(i)
    
    # 移动平均特征
    print("    - 创建滚动窗口特征 (7, 14, 30天)")
    for window in [7, 14, 30]:
        df_copy[f'rolling_mean_{window}'] = df_copy[target_col].rolling(window=window, min_periods=1).mean().shift(1)
        df_copy[f'rolling_std_{window}'] = df_copy[target_col].rolling(window=window, min_periods=1).std().shift(1)
    
    # 差分和趋势特征
    print("    - 创建差分和趋势特征")
    df_copy['diff_1'] = df_copy[target_col].diff().shift(1)
    df_copy['diff_7'] = df_copy[target_col].diff(7).shift(1)
    
    df_copy['trend'] = df_copy[target_col].rolling(window=7, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    ).shift(1)
    
    # 处理缺失值
    initial_rows = len(df_copy)
    df_copy.dropna(inplace=True)
    print(f"    - 特征工程导致数据行数从 {initial_rows} 减少到 {len(df_copy)}")
    
    return df_copy

def prepare_data_legacy(df, sequence_length, train_ratio=0.7, val_ratio=0.15):
    """
    MODIFIED: This function now perfectly mirrors the data prep logic from the
    original script, including the critical step of truncating the flat data
    to align with the sequence data.
    """
    print("\n--- 正在使用旧版兼容模式准备数据 (70/15/15 分割) ---")
    
    # 1. 创建高级特征
    df_features = create_advanced_features(df, target_col='y')
    feature_cols = [c for c in df_features.columns if c not in ['unique_id', 'ds', 'y']]
    
    X = df_features[feature_cols].values
    y = df_features['y'].values
    
    # 2. 数据分割点
    total_size = len(X)
    train_end = int(total_size * train_ratio)
    val_end = train_end + int(total_size * val_ratio)
    
    # 3. 数据缩放 (仅用训练集拟合)
    X_scaler = RobustScaler().fit(X[:train_end])
    y_scaler = RobustScaler().fit(y[:train_end].reshape(-1, 1))
    
    X_scaled = X_scaler.transform(X)
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).flatten()
    
    # 4. 分割完整数据集
    X_train_s, y_train_s = X_scaled[:train_end], y_scaled[:train_end]
    X_val_s, y_val_s = X_scaled[train_end:val_end], y_scaled[train_end:val_end]
    X_test_s, y_test_s = X_scaled[val_end:], y_scaled[val_end:]
    
    # --- 为不同类型的模型创建对齐的数据 ---
    
    # 5. "Sequence" 数据 (用于EnhancedNN, Transformer)
    X_train_seq, y_train_seq = create_sequences(X_train_s, y_train_s, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val_s, y_val_s, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_s, y_test_s, sequence_length)
    
    data_seq = { 'train': (X_train_seq, y_train_seq), 'val': (X_val_seq, y_val_seq), 'test': (X_test_seq, y_test_seq) }

    # 6. "Flat" 数据 (用于XGBoost, RF, SimpleNN等) - 进行关键的截断对齐
    y_train_flat = y_train_s[sequence_length:]
    y_val_flat = y_val_s[sequence_length:]
    y_test_flat = y_test_s[sequence_length:]

    X_train_flat = X_train_s[sequence_length:]
    X_val_flat = X_val_s[sequence_length:]
    X_test_flat = X_test_s[sequence_length:]
    
    # 确保对齐
    assert np.array_equal(y_train_flat, y_train_seq)
    assert np.array_equal(y_val_flat, y_val_seq)
    assert np.array_equal(y_test_flat, y_test_seq)
    
    data_flat = { 'train': (X_train_flat, y_train_flat), 'val': (X_val_flat, y_val_flat), 'test': (X_test_flat, y_test_flat) }

    # 7. 原始测试集目标值，用于最终评估 (使用对齐后的版本)
    y_test_orig_aligned = df_features['y'].values[val_end+sequence_length:]
    assert len(y_test_orig_aligned) == len(y_test_flat)
    
    data = {
        'flat': data_flat,
        'sequence': data_seq,
        'scalers': (X_scaler, y_scaler),
        'feature_cols': feature_cols,
        'y_test_orig': y_test_orig_aligned, # 使用对齐后的真实值
    }
    return data

# ==============================================================================
# 3. MODEL DEFINITIONS
# ==============================================================================
class SimpleNN(nn.Module):
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
        return self.fc3(x).squeeze(-1)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
    def forward(self, hidden_states):
        energy = torch.tanh(self.attn(hidden_states))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(hidden_states.size(0), 1).unsqueeze(1)
        attention_scores = torch.bmm(v, energy).squeeze(1)
        return F.softmax(attention_scores, dim=1)

class EnhancedNN(nn.Module):
    """LSTM + Attention, 适用于原始序列或带特征的序列"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(EnhancedNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = Attention(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        x = F.relu(self.fc1(context_vector))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)

class PositionalEncoding(nn.Module):
    """适用于batch_first=True的Transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=256, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(d_model, 1)
    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return self.decoder(output[:, -1, :]).squeeze(-1)

# ==============================================================================
# 4. UTILITY FUNCTIONS (DATA PREP, TRAINING LOOPS)
# ==============================================================================
def create_sequences(X, y, sequence_length=14):
    """
    MODIFIED: Re-added the 'sequence_length' parameter to match the original
    `archive/model_comparison.py` function signature.
    """
    xs, ys = [], []
    for i in range(len(X) - sequence_length):
        xs.append(X[i:(i + sequence_length)])
        ys.append(y[i + sequence_length])
    return np.array(xs), np.array(ys)

def universal_nn_trainer(model, train_ds, val_ds, model_name, epochs=100, batch_size=32, lr=0.001, patience=20):
    """
    MODIFIED: Switched to HuberLoss and fixed tensor shape issues.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.HuberLoss() # 使用HuberLoss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=False)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch.float())
            # 确保y_pred和y_batch形状一致
            loss = criterion(y_pred.view(-1), y_batch.view(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch.float())
                val_loss += criterion(y_val_pred.view(-1), y_val_batch.view(-1)).item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"  提前停止于 epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    return model

# ==============================================================================
# 5. MODEL RUNNERS
# ==============================================================================
def run_prophet(train_df, test_df):
    """
    MODIFIED: Performs a single-step forecast for each timestamp in the test set.
    """
    print("\n--- 运行 Prophet (单步预测模式) ---")
    model = Prophet()
    model.fit(train_df)
    future_df = test_df[['ds']] # Use test set timestamps for prediction
    forecast = model.predict(future_df)
    return forecast['yhat'].values

def run_arima(train_series, test_series):
    """
    MODIFIED: Performs a true rolling single-step forecast.
    This is computationally intensive as it re-fits the model at each step.
    """
    print("\n--- 运行 ARIMA (滚动单步预测模式) ---")
    print("  (这可能需要一些时间，因为它在测试集的每一步都重新拟合模型以获得最准确的单步预测)")
    history = list(train_series)
    predictions = []
    for t in range(len(test_series)):
        # A standard ARIMA order for non-seasonal data
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        # Add the actual observed value to the history for the next forecast
        obs = test_series[t]
        history.append(obs)
    return np.array(predictions)

def run_ensemble_models(X_train, y_train, X_test):
    print("\n--- 运行集成模型 (XGBoost, LightGBM, RandomForest) ---")
    predictions = {}
    
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    predictions['XGBoost'] = xgb_model.predict(X_test)

    lgb_model = lgb.LGBMRegressor(random_state=42)
    lgb_model.fit(X_train, y_train)
    predictions['LightGBM'] = lgb_model.predict(X_test)

    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    predictions['RandomForest'] = rf_model.predict(X_test)
    
    return predictions

def run_end_to_end_nn(train_series, test_series, sequence_length):
    # This function will be replaced by the new data prep logic
    pass

# ==============================================================================
# 6. EVALUATION & VISUALIZATION
# ==============================================================================
def evaluate_and_print(y_true, y_pred, model_name):
    """评估单个模型在单个序列上的表现"""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  - {model_name}: R²={r2:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse}

def plot_model_comparison(results, y_true, predictions, dataset_name, series_id):
    """绘制所有模型的预测结果与真实值的对比图"""
    # 解决中文和负号显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(20, 10))
    test_index = np.arange(len(y_true))
    plt.plot(test_index, y_true, label='真实值', color='black', linewidth=2.5, linestyle=':')
    
    sorted_models = sorted(results.items(), key=lambda item: item[1]['R2'], reverse=True)
    
    for model_name, result in sorted_models:
        if model_name in predictions:
            r2 = result['R2']
            plt.plot(test_index, predictions[model_name], label=f"{model_name} (R²: {r2:.3f})", alpha=0.8)
        
    plt.title(f'各模型在 {dataset_name} (序列: {series_id}) 上的单步预测对比', fontsize=20)
    plt.xlabel('预测时间步 (测试集)', fontsize=15)
    plt.ylabel('值', fontsize=15)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    save_path = f"plots/{dataset_name}_{series_id}_model_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n模型预测对比图已保存为 '{save_path}'")

# ==============================================================================
# 7. MAIN ORCHESTRATOR
# ==============================================================================
def main():
    # --- CONFIGURATION ---
    DATASET_NAME = 'min_daily_temps' 
    SEQUENCE_LENGTH = 14
    
    master_df = load_dataset(DATASET_NAME)
    if master_df is None: return

    series_id = master_df['unique_id'].unique()[0]
    print(f"\n{'='*20} 正在处理序列: {series_id} {'='*20}")
    
    # --- 1. 使用旧版兼容模式准备所有数据 ---
    data = prepare_data_legacy(master_df, sequence_length=SEQUENCE_LENGTH)
    
    # 解包所有需要的数据
    X_train_flat, y_train_flat = data['flat']['train']
    X_val_flat, y_val_flat = data['flat']['val']
    X_test_flat, y_test_flat = data['flat']['test']
    
    X_train_seq, y_train_seq = data['sequence']['train']
    X_val_seq, y_val_seq = data['sequence']['val']
    X_test_seq, y_test_seq = data['sequence']['test']

    X_scaler, y_scaler = data['scalers']
    y_test_orig = data['y_test_orig'] # The one true y_test for all models
    
    all_predictions = {}
    all_results = {}

    # --- 2. 训练所有模型 ---
    print("  - 暂时跳过 Prophet & ARIMA 以加速特征工程验证")

    # Ensemble Models with specific hyperparameters from the original script
    print("\n--- 训练集成模型 (使用旧版超参数) ---")
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42).fit(X_train_flat, y_train_flat)
    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.01, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42).fit(X_train_flat, y_train_flat)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_split=5, min_samples_leaf=2, random_state=42).fit(X_train_flat, y_train_flat)

    # NN Models
    print("\n--- 训练神经网络模型 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SimpleNN
    print("  训练 SimpleNN...")
    snn_model = SimpleNN(X_train_flat.shape[1])
    train_ds_flat = TensorDataset(torch.from_numpy(X_train_flat).float(), torch.from_numpy(y_train_flat).float())
    val_ds_flat = TensorDataset(torch.from_numpy(X_val_flat).float(), torch.from_numpy(y_val_flat).float())
    snn_model = universal_nn_trainer(snn_model, train_ds_flat, val_ds_flat, 'SimpleNN', patience=20)

    # EnhancedNN
    print("  训练 EnhancedNN...")
    enn_model = EnhancedNN(input_size=X_train_seq.shape[2])
    train_ds_seq = TensorDataset(torch.from_numpy(X_train_seq).float(), torch.from_numpy(y_train_seq).float())
    val_ds_seq = TensorDataset(torch.from_numpy(X_val_seq).float(), torch.from_numpy(y_val_seq).float())
    enn_model = universal_nn_trainer(enn_model, train_ds_seq, val_ds_seq, 'EnhancedNN', lr=0.0001, patience=20)
    
    # Transformer
    print("  训练 Transformer...")
    tf_model = TimeSeriesTransformer(input_dim=X_train_seq.shape[2])
    tf_model = universal_nn_trainer(tf_model, train_ds_seq, val_ds_seq, 'Transformer', lr=0.0001, patience=20)

    # --- 3. 评估所有模型 ---
    print("\n" + "="*30 + " 最终评估结果 " + "="*30)
    
    # Evaluate all models against the same y_test_orig
    
    # Ensemble predictions
    for name, model in {'XGBoost': xgb_model, 'LightGBM': lgb_model, 'RandomForest': rf_model}.items():
        preds_scaled = model.predict(X_test_flat)
        preds_orig = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        all_predictions[name] = preds_orig
        all_results[name] = evaluate_and_print(y_test_orig, preds_orig, name)
        
    # SimpleNN predictions
    snn_model.eval()
    with torch.no_grad():
        snn_preds_s = snn_model(torch.from_numpy(X_test_flat).float().to(device)).cpu().detach().numpy()
    snn_preds_orig = y_scaler.inverse_transform(snn_preds_s.reshape(-1, 1)).flatten()
    all_predictions['SimpleNN'] = snn_preds_orig
    all_results['SimpleNN'] = evaluate_and_print(y_test_orig, snn_preds_orig, 'SimpleNN')

    # Sequence models predictions
    enn_model.eval()
    with torch.no_grad():
        enn_preds_s = enn_model(torch.from_numpy(X_test_seq).float().to(device)).cpu().detach().numpy()
    enn_preds_orig = y_scaler.inverse_transform(enn_preds_s.reshape(-1, 1)).flatten()
    all_predictions['EnhancedNN'] = enn_preds_orig
    all_results['EnhancedNN'] = evaluate_and_print(y_test_orig, enn_preds_orig, 'EnhancedNN')
    
    tf_model.eval()
    with torch.no_grad():
        tf_preds_s = tf_model(torch.from_numpy(X_test_seq).float().to(device)).cpu().detach().numpy()
    tf_preds_orig = y_scaler.inverse_transform(tf_preds_s.reshape(-1, 1)).flatten()
    all_predictions['Transformer'] = tf_preds_orig
    all_results['Transformer'] = evaluate_and_print(y_test_orig, tf_preds_orig, 'Transformer')

    # --- 4. 可视化 ---
    plot_model_comparison(all_results, y_test_orig, all_predictions, DATASET_NAME, series_id)

if __name__ == '__main__':
    main() 