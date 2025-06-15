import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
from mvse_embedding import MVSEEmbedding

warnings.filterwarnings('ignore')


class MVSEProbeForecaster(nn.Module):
    """
    基于 MVSEEmbedding 的时间序列预测模型
    
    结合了多视角序列编码和传统的滞后特征
    """
    
    def __init__(self, input_dim, mvse_d_hidden=128, mvse_d_out=32, 
                 num_lags=14, mask_rate=0.3, final_hidden=64):
        super(MVSEProbeForecaster, self).__init__()
        
        self.input_dim = input_dim
        self.num_lags = num_lags
        self.mvse_d_out = mvse_d_out
        
        # MVSE 编码器：将历史序列编码为全局特征
        self.mvse_encoder = MVSEEmbedding(
            d_input=input_dim,
            d_hidden=mvse_d_hidden,
            d_out=mvse_d_out,
            mask_rate=mask_rate
        )
        
        # 最终预测层：结合 MVSE 特征和滞后特征
        total_features = mvse_d_out + num_lags  # MVSE特征 + 滞后特征
        self.predictor = nn.Sequential(
            nn.Linear(total_features, final_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(final_hidden, final_hidden // 2),
            nn.ReLU(),
            nn.Linear(final_hidden // 2, 1)
        )
        
    def forward(self, hist_seq, lag_features):
        """
        前向传播
        
        Args:
            hist_seq (torch.Tensor): 历史序列 (B, T, D)
            lag_features (torch.Tensor): 滞后特征 (B, num_lags)
            
        Returns:
            torch.Tensor: 预测结果 (B, 1)
        """
        # 1. 使用 MVSE 编码历史序列
        mvse_features = self.mvse_encoder(hist_seq)  # (B, mvse_d_out)
        
        # 2. 拼接 MVSE 特征和滞后特征
        combined_features = torch.cat([mvse_features, lag_features], dim=1)  # (B, total_features)
        
        # 3. 最终预测
        prediction = self.predictor(combined_features)  # (B, 1)
        
        return prediction


def create_mvse_probe_features(df, target_col='temp', hist_len=90, num_lags=14):
    """
    为 MVSEProbeForecaster 创建特征
    
    Args:
        df (pd.DataFrame): 输入数据
        target_col (str): 目标列名
        hist_len (int): 历史序列长度
        num_lags (int): 滞后特征数量
        
    Returns:
        tuple: (hist_sequences, lag_features, targets, valid_indices)
    """
    print(f"🔧 创建 MVSE 探针特征...")
    print(f"   - 历史序列长度: {hist_len}")
    print(f"   - 滞后特征数量: {num_lags}")
    
    # 确保有必要的时间特征
    df_copy = df.copy()
    if 'dayofweek' not in df_copy:
        df_copy['dayofweek'] = df_copy['date'].dt.dayofweek
    if 'month' not in df_copy:
        df_copy['month'] = df_copy['date'].dt.month
    
    # 标准化目标变量
    target_scaler = MinMaxScaler()
    df_copy['temp_scaled'] = target_scaler.fit_transform(df_copy[[target_col]])
    
    # 准备多维输入特征 (temp, dayofweek, month)
    feature_cols = ['temp_scaled', 'dayofweek', 'month']
    
    hist_sequences = []
    lag_features = []
    targets = []
    valid_indices = []
    
    # 滑动窗口创建序列
    for i in range(hist_len + num_lags, len(df_copy)):
        # 1. 历史序列 (用于 MVSE 编码)
        hist_start = i - hist_len - num_lags
        hist_end = i - num_lags
        hist_seq = df_copy.iloc[hist_start:hist_end][feature_cols].values
        hist_sequences.append(hist_seq)
        
        # 2. 滞后特征 (最近的 num_lags 个值)
        lag_start = i - num_lags
        lag_end = i
        lag_vals = df_copy.iloc[lag_start:lag_end]['temp_scaled'].values
        lag_features.append(lag_vals)
        
        # 3. 目标值 (当前时刻)
        target_val = df_copy.iloc[i]['temp_scaled']
        targets.append(target_val)
        
        # 4. 记录有效索引
        valid_indices.append(df_copy.index[i])
    
    hist_sequences = np.array(hist_sequences, dtype=np.float32)
    lag_features = np.array(lag_features, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    print(f"✅ 特征创建完成:")
    print(f"   - 历史序列形状: {hist_sequences.shape}")
    print(f"   - 滞后特征形状: {lag_features.shape}")
    print(f"   - 目标值形状: {targets.shape}")
    
    return hist_sequences, lag_features, targets, valid_indices, target_scaler


def train_mvse_probe_model(hist_sequences, lag_features, targets, 
                          input_dim=3, epochs=100, batch_size=32, 
                          learning_rate=0.001, mask_rate=0.3):
    """
    训练 MVSEProbeForecaster 模型
    """
    print(f"🚀 开始训练 MVSEProbeForecaster...")
    
    # 数据分割
    train_size = int(len(hist_sequences) * 0.8)
    
    X_hist_train = hist_sequences[:train_size]
    X_lag_train = lag_features[:train_size]
    y_train = targets[:train_size]
    
    X_hist_val = hist_sequences[train_size:]
    X_lag_val = lag_features[train_size:]
    y_val = targets[train_size:]
    
    # 转换为张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_hist_train = torch.FloatTensor(X_hist_train).to(device)
    X_lag_train = torch.FloatTensor(X_lag_train).to(device)
    y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    X_hist_val = torch.FloatTensor(X_hist_val).to(device)
    X_lag_val = torch.FloatTensor(X_lag_val).to(device)
    y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    # 创建模型
    model = MVSEProbeForecaster(
        input_dim=input_dim,
        mask_rate=mask_rate,
        num_lags=lag_features.shape[1]
    ).to(device)
    
    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练设置
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # 训练循环
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        
        # 简单的批次训练
        for i in range(0, len(X_hist_train), batch_size):
            end_idx = min(i + batch_size, len(X_hist_train))
            
            batch_hist = X_hist_train[i:end_idx]
            batch_lag = X_lag_train[i:end_idx]
            batch_y = y_train[i:end_idx]
            
            optimizer.zero_grad()
            predictions = model(batch_hist, batch_lag)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / (len(X_hist_train) // batch_size + 1)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_hist_val, X_lag_val)
            val_loss = criterion(val_predictions, y_val).item()
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"   早停触发！最佳验证损失: {best_val_loss:.6f}")
            break
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    return model, best_val_loss


def generate_mvse_probe_features_for_tlafs(df, target_col='temp'):
    """
    为 T-LAFS 框架生成 MVSE 探针特征
    
    这个函数可以集成到现有的 T-LAFS 系统中
    """
    print("🔮 生成 MVSE 探针特征用于 T-LAFS...")
    
    # 创建特征
    hist_sequences, lag_features, targets, valid_indices, target_scaler = create_mvse_probe_features(
        df, target_col=target_col, hist_len=90, num_lags=14
    )
    
    # 训练模型
    model, best_loss = train_mvse_probe_model(
        hist_sequences, lag_features, targets,
        epochs=50, mask_rate=0.3
    )
    
    # 生成特征
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        # 获取 MVSE 编码特征
        hist_tensor = torch.FloatTensor(hist_sequences).to(device)
        mvse_features = model.mvse_encoder(hist_tensor).cpu().numpy()
        
        # 获取池化特征用于分析
        pooling_features = model.mvse_encoder.get_pooling_features(hist_tensor)
        gap_features = pooling_features['gap'].cpu().numpy()
        gmp_features = pooling_features['gmp'].cpu().numpy()
        masked_gap_features = pooling_features['masked_gap'].cpu().numpy()
    
    # 创建特征 DataFrame
    feature_names = []
    all_features = []
    
    # 1. MVSE 主要特征
    mvse_cols = [f"mvse_feat_{i}" for i in range(mvse_features.shape[1])]
    feature_names.extend(mvse_cols)
    all_features.append(mvse_features)
    
    # 2. 池化特征的统计摘要
    gap_stats = np.column_stack([
        gap_features.mean(axis=1),
        gap_features.std(axis=1),
        gap_features.max(axis=1),
        gap_features.min(axis=1)
    ])
    gap_stat_cols = ['mvse_gap_mean', 'mvse_gap_std', 'mvse_gap_max', 'mvse_gap_min']
    feature_names.extend(gap_stat_cols)
    all_features.append(gap_stats)
    
    gmp_stats = np.column_stack([
        gmp_features.mean(axis=1),
        gmp_features.std(axis=1),
        gmp_features.max(axis=1),
        gmp_features.min(axis=1)
    ])
    gmp_stat_cols = ['mvse_gmp_mean', 'mvse_gmp_std', 'mvse_gmp_max', 'mvse_gmp_min']
    feature_names.extend(gmp_stat_cols)
    all_features.append(gmp_stats)
    
    # 合并所有特征
    final_features = np.concatenate(all_features, axis=1)
    
    # 创建 DataFrame
    features_df = pd.DataFrame(
        final_features,
        index=valid_indices,
        columns=feature_names
    )
    
    print(f"✅ MVSE 探针特征生成完成:")
    print(f"   - 特征数量: {len(feature_names)}")
    print(f"   - 样本数量: {len(features_df)}")
    print(f"   - 训练损失: {best_loss:.6f}")
    
    # 使用 shift(1) 避免数据泄露
    return df.join(features_df.shift(1))


def test_mvse_integration():
    """
    测试 MVSE 集成功能
    """
    print("🧪 测试 MVSE 集成功能...")
    
    # 创建模拟数据
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    np.random.seed(42)
    
    # 生成带有季节性和趋势的时间序列
    t = np.arange(len(dates))
    seasonal = 10 * np.sin(2 * np.pi * t / 365.25)  # 年度季节性
    trend = 0.01 * t  # 线性趋势
    noise = np.random.normal(0, 2, len(dates))
    temp = 20 + seasonal + trend + noise
    
    df = pd.DataFrame({
        'date': dates,
        'temp': temp
    })
    
    print(f"📊 模拟数据: {len(df)} 个样本")
    
    # 测试 MVSE 特征生成
    df_with_mvse = generate_mvse_probe_features_for_tlafs(df, target_col='temp')
    
    # 检查结果
    mvse_cols = [col for col in df_with_mvse.columns if 'mvse_' in col]
    print(f"🎯 生成的 MVSE 特征: {len(mvse_cols)} 个")
    print(f"   前5个特征: {mvse_cols[:5]}")
    
    # 简单的性能测试
    from sklearn.ensemble import RandomForestRegressor
    
    # 准备数据
    feature_cols = mvse_cols
    df_clean = df_with_mvse.dropna()
    
    if len(df_clean) > 100 and len(feature_cols) > 0:
        X = df_clean[feature_cols]
        y = df_clean['temp']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # 训练随机森林
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        print(f"🎯 MVSE 特征性能测试:")
        print(f"   - R² 得分: {r2:.4f}")
        print(f"   - 特征重要性前5: {sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: x[1], reverse=True)[:5]}")
    
    print("\n✅ MVSE 集成测试完成！")


if __name__ == "__main__":
    test_mvse_integration() 