import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import os
import math
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class AgentAttentionProbe(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_agents: int):
        super().__init__()
        self.num_agents = num_agents
        self.agents = nn.Parameter(torch.randn(1, num_agents, d_model))
        self.pos_encoder = PositionalEncoding(d_model)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

    def forward(self, x_embedded):
        x_pos = self.pos_encoder(x_embedded)
        agent_queries = self.agents.expand(x_embedded.shape[0], -1, -1)
        attn_output, _ = self.cross_attention(query=agent_queries, key=x_pos, value=x_pos)
        return attn_output

class ProbeForecaster(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_agents: int, num_lags: int, num_exog: int):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.attention_probe = AgentAttentionProbe(d_model, nhead, num_agents)
        self.forecasting_head = nn.Sequential(
            nn.Linear(d_model * num_agents + num_lags + num_exog, 512),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x, x_lags, x_exog):
        x_embedded = self.input_embedding(x)
        probe_output = self.attention_probe(x_embedded)
        probe_output_flat = probe_output.reshape(probe_output.size(0), -1)
        combined_features = torch.cat([probe_output_flat, x_lags, x_exog], dim=1)
        return self.forecasting_head(combined_features)

def get_data(file_path='data/total_cleaned.csv'):
    df = pd.read_csv(file_path)
    if '日期' in df.columns:
        df.rename(columns={'日期': 'date', '成交商品件数': 'temp'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    return df.reset_index(drop=True)

def create_sequences_and_lags(data, hist_len, num_lags):
    X_hist, y_target, X_lags, X_exog = [], [], [], []
    if len(data) < hist_len + num_lags: return None, None, None, None
    for i in range(num_lags, len(data) - hist_len):
        X_hist.append(data.iloc[i : i + hist_len][['temp', 'dayofweek', 'month']].values)
        y_target.append(data.iloc[i + hist_len]['temp'])
        X_lags.append(data.iloc[i + hist_len - num_lags : i + hist_len]['temp'].values)
        X_exog.append(data.iloc[i + hist_len][['dayofweek', 'month']].values)
    return np.array(X_hist), np.array(y_target).reshape(-1, 1), np.array(X_lags), np.array(X_exog)

def train_probe_forecaster(config, model_save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = get_data()
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)
    
    target_scaler = MinMaxScaler()
    train_df['temp'] = target_scaler.fit_transform(train_df[['temp']])
    val_df['temp'] = target_scaler.transform(val_df[['temp']])

    X_train, y_train, X_train_lags, X_train_exog = create_sequences_and_lags(train_df, config['seq_len'], config['num_lags'])
    X_val, y_val, X_val_lags, X_val_exog = create_sequences_and_lags(val_df, config['seq_len'], config['num_lags'])
    
    if X_train is None: return
    
    # --- 强制转换为 float32，避免 object 类型导致的张量构造错误 ---
    X_train = X_train.astype(np.float32)
    X_train_lags = X_train_lags.astype(np.float32)
    X_train_exog = X_train_exog.astype(np.float32)
    y_train = y_train.astype(np.float32)

    X_val = X_val.astype(np.float32)
    X_val_lags = X_val_lags.astype(np.float32)
    X_val_exog = X_val_exog.astype(np.float32)
    y_val = y_val.astype(np.float32)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(X_train_lags), torch.from_numpy(X_train_exog), torch.from_numpy(y_train)), batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(X_val_lags), torch.from_numpy(X_val_exog), torch.from_numpy(y_val)), batch_size=config['batch_size'])

    model = ProbeForecaster(X_train.shape[2], config['d_model'], config['nhead'], config['num_agents'], config['num_lags'], X_train_exog.shape[1]).to(device)
    optimizer, criterion = optim.Adam(model.parameters(), lr=config['learning_rate']), nn.MSELoss()
    
    best_val_loss, epochs_no_improve, best_model_state = float('inf'), 0, None
    train_losses, val_losses = [], []

    print(f"🚀 开始训练 ProbeForecaster，最多 {config['epochs']} 轮...")
    print(f"📊 训练集: {len(train_loader)} 批次, 验证集: {len(val_loader)} 批次")
    
    # 使用tqdm添加进度条
    pbar = tqdm(range(config['epochs']), desc="训练进度")
    
    for epoch in pbar:
        # 训练阶段
        model.train()
        train_loss = 0
        for hist, lags, exog, targets in train_loader:
            optimizer.zero_grad()
            preds = model(hist.to(device), lags.to(device), exog.to(device))
            loss = criterion(preds, targets.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for hist, lags, exog, targets in val_loader:
                val_loss += criterion(model(hist.to(device), lags.to(device), exog.to(device)), targets.to(device)).item()
        
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 更新进度条显示
        pbar.set_postfix({
            'Train Loss': f'{avg_train_loss:.6f}',
            'Val Loss': f'{avg_val_loss:.6f}',
            'Best': f'{best_val_loss:.6f}'
        })
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss, epochs_no_improve, best_model_state = avg_val_loss, 0, model.state_dict()
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= config['patience']:
            print(f"\n⏹️  早停触发！已连续 {config['patience']} 轮无改善")
            break
    
    pbar.close()
    
    # 保存最佳模型
    if best_model_state:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(best_model_state, model_save_path)
        print(f"✅ 最佳模型已保存到: {model_save_path}")
        print(f"🎯 最佳验证损失: {best_val_loss:.6f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', color='blue', alpha=0.7)
    plt.plot(val_losses, label='验证损失', color='red', alpha=0.7)
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('损失 (Loss)')
    plt.title('ProbeForecaster 训练曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制预测效果示例
    plt.subplot(1, 2, 2)
    model.load_state_dict(best_model_state)
    model.eval()
    
    # 取一个验证批次进行可视化
    with torch.no_grad():
        for hist, lags, exog, targets in val_loader:
            preds = model(hist.to(device), lags.to(device), exog.to(device))
            
            # 反标准化
            targets_orig = target_scaler.inverse_transform(targets.cpu().numpy())
            preds_orig = target_scaler.inverse_transform(preds.cpu().numpy())
            
            # 只显示前20个样本
            n_show = min(20, len(targets_orig))
            plt.scatter(targets_orig[:n_show], preds_orig[:n_show], alpha=0.6, color='green')
            break
    
    # 绘制理想预测线
    min_val = min(targets_orig[:n_show].min(), preds_orig[:n_show].min())
    max_val = max(targets_orig[:n_show].max(), preds_orig[:n_show].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='理想预测')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('ProbeForecaster 预测效果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = model_save_path.replace('.pth', '_training_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 训练结果图已保存到: {plot_path}")
    plt.show()
    
    return model, target_scaler

def generate_probe_features(df, config, model_path):
    """
    生成高质量的探针特征，结合感知损失训练的优势
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProbeForecaster(3, config['d_model'], config['nhead'], config['num_agents'], config['num_lags'], 2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    df_copy = df.copy()
    if 'dayofweek' not in df_copy: df_copy['dayofweek'] = df_copy['date'].dt.dayofweek
    if 'month' not in df_copy: df_copy['month'] = df_copy['date'].dt.month
    
    temp_scaler = MinMaxScaler()
    df_copy['temp_scaled'] = temp_scaler.fit_transform(df_copy[['temp']])
    
    all_embeddings, all_predictions, valid_indices = [], [], []
    hist_len, num_lags = config['seq_len'], config['num_lags']
    
    for i in range(num_lags, len(df_copy) - hist_len):
        hist_values = df_copy.iloc[i : i + hist_len][['temp_scaled', 'dayofweek', 'month']].values.astype(np.float32)
        hist_tensor = torch.from_numpy(hist_values).unsqueeze(0).to(device)
        
        # 获取滞后特征和外生特征
        lag_values = df_copy.iloc[i + hist_len - num_lags : i + hist_len]['temp_scaled'].values.astype(np.float32)
        exog_values = df_copy.iloc[i + hist_len][['dayofweek', 'month']].values.astype(np.float32)
        
        lag_tensor = torch.from_numpy(lag_values).unsqueeze(0).to(device)
        exog_tensor = torch.from_numpy(exog_values).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 1. 获取注意力探针的嵌入（包含位置编码）
            x_embedded = model.input_embedding(hist_tensor)
            probe_output = model.attention_probe(x_embedded)  # 已包含位置编码
            
            # 2. 获取完整的预测结果（利用感知损失训练的优势）
            prediction = model(hist_tensor, lag_tensor, exog_tensor)
            
            # 3. 只保留最重要的探针特征（降维）
            # 计算每个agent的平均激活作为代表性特征
            probe_summary = probe_output.mean(dim=1)  # (1, d_model)
            
        all_embeddings.append(probe_summary.cpu().numpy())
        all_predictions.append(prediction.cpu().numpy())
        valid_indices.append(df_copy.index[i + hist_len])
    
    if not all_embeddings: return df

    # 创建高质量特征集
    embeddings_array = np.concatenate(all_embeddings)  # (n_samples, d_model)
    predictions_array = np.concatenate(all_predictions)  # (n_samples, 1)
    
    # 生成特征名称
    embed_cols = [f"probe_embed_{j}" for j in range(embeddings_array.shape[1])]
    pred_cols = ["probe_prediction"]
    
    # 组合特征
    all_features = np.concatenate([embeddings_array, predictions_array], axis=1)
    all_col_names = embed_cols + pred_cols
    
    features_df = pd.DataFrame(all_features, index=valid_indices, columns=all_col_names)
    
    # 添加一些基于探针特征的衍生特征
    features_df['probe_embed_mean'] = embeddings_array.mean(axis=1)
    features_df['probe_embed_std'] = embeddings_array.std(axis=1)
    features_df['probe_embed_max'] = embeddings_array.max(axis=1)
    features_df['probe_embed_min'] = embeddings_array.min(axis=1)
    
    # 使用shift(1)避免数据泄露
    return df.join(features_df.shift(1))