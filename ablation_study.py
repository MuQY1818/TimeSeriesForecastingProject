import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score

# 从主脚本中导入必要的函数和类
from model_comparison import (
    prepare_data, 
    SimpleNN, 
    EnhancedNN, 
    train_neural_network, 
    get_feature_groups
)

def plot_focused_ablation_results(results, save_path):
    """可视化专注消融实验的结果"""
    # 修正: 创建DataFrame时，明确指定index
    df_plot = pd.DataFrame(results, index=[0]).T
    df_plot.columns = ['R² Score']
    df_plot = df_plot.sort_values(by='R² Score', ascending=False)
    
    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(df_plot)))
    
    fig, ax = plt.subplots(figsize=(14, 10))
    bars = ax.barh(df_plot.index, df_plot['R² Score'], color=colors, height=0.6)
    
    # --- 美化 ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='y', color='#EEEEEE', linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    ax.set_title('精细化特征消融实验 (Focused Ablation Study)', fontsize=20, pad=20, weight='bold')
    ax.set_xlabel('R² 分数', fontsize=15, labelpad=15)

    ax.set_ylabel('实验', labelpad=15, fontsize=16, color='#333333')
    ax.tick_params(axis='y', labelsize=12, colors='#555555')

    ax.legend(frameon=False, loc='upper right', fontsize=12)

    def autolabel(rects):
        for i, rect in enumerate(rects):
            width = rect.get_width()
            ax.annotate(f'{width:.3f}',
                        xy=(width, rect.get_y() + rect.get_height() / 2),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha='left', va='center',
                        fontsize=11, color='#333333', weight='semibold')

    autolabel(bars)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFocused ablation study plot saved to {save_path}")
    plt.close(fig)


def run_focused_ablation_study(df, sequence_length):
    """运行一系列精确的、集中的特征消融实验，并生成对比图"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    base_path = 'models'
    os.makedirs(base_path, exist_ok=True)

    log_file_path = os.path.join(base_path, 'ablation_training_log.csv')
    if os.path.exists(log_file_path):
        os.remove(log_file_path) # 每次运行时重新创建日志

    print("="*80)
    print(" 开始运行专注特征消融实验 (Focused Feature Ablation Study) ".center(80, "="))
    print("="*80)

    results = {}

    # --- SimpleNN 崩溃点探索 ---
    print("\n--- 实验: SimpleNN 崩溃点探索 ---")
    
    # 基线：移除在之前实验中被证明对SimpleNN最重要的特征
    base_features_to_remove = ['diff_trend', 'lag']

    # 在此基础上，进一步移除 'rolling' 组的子特征来定位引爆点
    breakdown_setups = {
        "移除 rolling_mean": ['rolling_mean'],
        "移除 rolling_std": ['rolling_std'],
        "移除 ma_ratio": ['ma_ratio'],
    }

    for name, extra_features_to_remove in breakdown_setups.items():
        print(f"\n  测试: {name}")
        features_to_remove = base_features_to_remove + extra_features_to_remove
        
        print(f"    移除的特征组: {features_to_remove}")
        data = prepare_data(df, sequence_length=sequence_length, excluded_features=features_to_remove)
        
        X_train, y_train = data['flat']['train']
        X_val, y_val = data['flat']['val']
        X_test, y_test = data['flat']['test']
        y_scaler = data['scalers'][1]
        
        print("    正在训练模型...")
        model = SimpleNN(input_size=X_train.shape[1])
        model = train_neural_network(
            model, X_train, y_train, X_val, y_val, 
            patience=15, epochs=300,
            log_file_path=log_file_path,
            ablation_setup_name=f'SimpleNN_breakdown_{name.replace(" ", "_")}',
            model_name='SimpleNN'
        )

        print("    正在评估模型...")
        model.eval()
        with torch.no_grad():
            predictions = model(torch.FloatTensor(X_test)).numpy().flatten()
        
        y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        preds_orig = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        r2 = r2_score(y_test_orig, preds_orig)

        results[f'SimpleNN\n(No diff/lag, {name})'] = r2
        print(f"    完成. R² score: {r2:.4f}")

    # --- 运行 EnhancedNN 的专项实验 ---
    print("\n\n--- 实验: EnhancedNN '毒性'特征验证 ---")
    
    print("  准备只含 'lag' 特征的数据集...")
    # 修正: 直接定义所有已知的特征组，并排除 'lag' 组
    all_known_groups = ['time', 'cyclical', 'lag', 'rolling_mean', 'rolling_std', 'ma_ratio', 'diff_trend']
    groups_to_exclude = [g for g in all_known_groups if g != 'lag']

    print(f"    将排除的特征组: {groups_to_exclude}")
    data_seq = prepare_data(df, sequence_length=sequence_length, excluded_features=groups_to_exclude)
    
    X_train_seq, y_train_seq = data_seq['sequence']['train']
    X_val_seq, y_val_seq = data_seq['sequence']['val']
    X_test_seq, y_test_seq = data_seq['sequence']['test']
    y_scaler_seq = data_seq['scalers'][1]

    print("  正在训练模型...")
    model = EnhancedNN(input_size=X_train_seq.shape[2])
    model = train_neural_network(
        model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
        lr=0.0001, patience=15, epochs=300,
        log_file_path=log_file_path,
        ablation_setup_name='EnhancedNN_Lag_Only',
        model_name='EnhancedNN'
    )
    
    print("  正在评估模型...")
    model.eval()
    with torch.no_grad():
        predictions_seq = model(torch.FloatTensor(X_test_seq)).numpy().flatten()
    
    y_test_orig_seq = y_scaler_seq.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    preds_orig_seq = y_scaler_seq.inverse_transform(predictions_seq.reshape(-1, 1)).flatten()
    r2 = r2_score(y_test_orig_seq, preds_orig_seq)

    results['EnhancedNN (Lag Only)'] = r2
    print(f"  完成. R² score: {r2:.4f}\n")

    # --- 结果可视化 ---
    print("\n实验全部完成，正在生成最终图表...")
    plot_focused_ablation_results(results, os.path.join(base_path, 'focused_ablation_study.png'))
    print(f"图表已保存至 {os.path.join(base_path, 'focused_ablation_study.png')}")
    print(f"详细训练日志已保存至 {log_file_path}")


def create_advanced_features(df, target_col='成交商品件数'):
    """这是一个辅助函数，直接从主脚本复制过来以确保一致性"""
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


if __name__ == "__main__":
    # 加载数据
    try:
        df_main = pd.read_csv('total_cleaned.csv')
        df_main['日期'] = pd.to_datetime(df_main['日期'])
        SEQUENCE_LENGTH = 14
        
        # 运行专注的消融实验
        run_focused_ablation_study(df_main, SEQUENCE_LENGTH)

    except FileNotFoundError:
        print("错误: 'total_cleaned.csv' not found.")
        print("请确保主脚本 `model_comparison.py` 已经运行过，或者数据集文件在本目录中。") 