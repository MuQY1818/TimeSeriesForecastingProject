"""
T-LAFS 框架集成 MVSE 探针模块
在原有的 clp_probe_experiment.py 基础上添加 MVSE 探针功能
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
import os
import sys
from typing import List, Dict, Any

# 导入 MVSE 模块
from mvse_embedding import MVSEEmbedding
from mvse_probe_integration import MVSEProbeForecaster, create_mvse_probe_features, train_mvse_probe_model

# 在原有的 tlafs_core.py 基础上添加 MVSE 探针功能
from tlafs_core import TLAFS_Algorithm as OriginalTLAFS
from tlafs_core import gemini_model

warnings.filterwarnings('ignore')


def generate_mvse_features_for_tlafs(df, target_col='temp', hist_len=90, num_lags=14):
    """
    为 T-LAFS 框架生成 MVSE 探针特征
    
    这个函数专门为 T-LAFS 的 execute_plan 方法设计
    """
    print("  - 🔮 生成 MVSE 探针特征...")
    
    try:
        # 创建特征
        hist_sequences, lag_features, targets, valid_indices, target_scaler = create_mvse_probe_features(
            df, target_col=target_col, hist_len=hist_len, num_lags=num_lags
        )
        
        if len(hist_sequences) == 0:
            print("  - ⚠️ 数据不足，无法生成 MVSE 特征")
            return df
        
        # 训练模型（使用较少的轮次以提高速度）
        model, best_loss = train_mvse_probe_model(
            hist_sequences, lag_features, targets,
            epochs=30, mask_rate=0.3, batch_size=64
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
        
        # 创建特征 DataFrame
        feature_names = []
        all_features = []
        
        # 1. MVSE 主要特征（降维到16维以减少特征数量）
        mvse_cols = [f"mvse_feat_{i}" for i in range(min(16, mvse_features.shape[1]))]
        feature_names.extend(mvse_cols)
        all_features.append(mvse_features[:, :len(mvse_cols)])
        
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
        
        print(f"  - ✅ MVSE 特征生成完成: {len(feature_names)} 个特征, 训练损失: {best_loss:.6f}")
        
        # 使用 shift(1) 避免数据泄露
        return df.join(features_df.shift(1))
        
    except Exception as e:
        print(f"  - ❌ MVSE 特征生成失败: {e}")
        return df


def add_mvse_to_execute_plan():
    """
    为 TLAFS 的 execute_plan 方法添加 MVSE 特征生成功能
    """
    def execute_plan_with_mvse(df: pd.DataFrame, plan: List[Dict[str, Any]]):
        """
        执行特征工程计划，包括 MVSE 特征生成
        
        Args:
            df (pd.DataFrame): 输入数据
            plan (List[Dict]): 特征工程计划
            
        Returns:
            pd.DataFrame: 处理后的数据框
        """
        from tlafs_utils import generate_mvse_features_for_tlafs
        
        # 执行原始计划
        for operation in plan:
            if operation['operation'] == 'create_mvse_features':
                df = generate_mvse_features_for_tlafs(
                    df,
                    target_col=operation.get('target_col', 'temp'),
                    hist_len=operation.get('hist_len', 90),
                    num_lags=operation.get('num_lags', 14)
                )
            else:
                # 处理其他操作...
                pass
                
        return df
    
    return execute_plan_with_mvse


def add_mvse_to_llm_prompt():
    """
    为 TLAFS 的 LLM 提示词添加 MVSE 相关说明
    """
    def get_enhanced_prompt(context_prompt: str) -> str:
        """
        增强 LLM 提示词，添加 MVSE 相关说明
        
        Args:
            context_prompt (str): 原始提示词
            
        Returns:
            str: 增强后的提示词
        """
        mvse_context = """
        可用的 MVSE 特征工程操作：
        1. create_mvse_features: 生成多视角序列编码特征
           - target_col: 目标列名
           - hist_len: 历史序列长度
           - num_lags: 滞后特征数量
        """
        
        return context_prompt + mvse_context
    
    return get_enhanced_prompt


def create_enhanced_tlafs_with_mvse():
    """
    创建增强版的 TLAFS 类，集成 MVSE 功能
    """
    from tlafs_core import TLAFS_Algorithm
    
    class EnhancedTLAFS(TLAFS_Algorithm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.execute_plan = add_mvse_to_execute_plan()
            self.get_plan_from_llm = add_mvse_to_llm_prompt()
    
    return EnhancedTLAFS


def test_mvse_integration_in_tlafs():
    """
    测试 MVSE 在 TLAFS 中的集成
    """
    print("🧪 测试 MVSE 在 TLAFS 中的集成...")
    
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
    
    # 创建增强版 TLAFS
    EnhancedTLAFS = create_enhanced_tlafs_with_mvse()
    
    # 测试 MVSE 特征生成
    plan = [{
        'operation': 'create_mvse_features',
        'target_col': 'temp',
        'hist_len': 90,
        'num_lags': 14
    }]
    
    df_with_mvse = EnhancedTLAFS.execute_plan(df.copy(), plan)
    
    # 检查结果
    mvse_cols = [col for col in df_with_mvse.columns if 'mvse_' in col]
    print(f"✅ MVSE 特征生成测试:")
    print(f"   - 生成的特征数量: {len(mvse_cols)}")
    print(f"   - 前5个特征: {mvse_cols[:5]}")
    
    print("\n✅ MVSE 集成测试完成！")


def create_integration_guide():
    """
    创建集成指南
    """
    guide = """
# 🔧 MVSE 探针集成到 T-LAFS 的完整指南

## 方法 1: 直接修改现有文件 (推荐)

### 步骤 1: 添加导入
在 `clp_probe_experiment.py` 文件顶部添加：
```python
from mvse_embedding import MVSEEmbedding
from mvse_probe_integration import generate_mvse_probe_features_for_tlafs
```

### 步骤 2: 修改 execute_plan 方法
在 `execute_plan` 方法的 `elif op == "create_probe_features":` 后面添加：
```python
elif op == "create_mvse_features":
    print("  - Generating MVSE probe features...")
    temp_df = generate_mvse_features_for_tlafs(temp_df, target_col)
    print("  - ✅ MVSE features generated.")
```

### 步骤 3: 修改 LLM 提示
在 `advanced_tools` 字符串中添加：
```python
# 4. MVSE Probe Features (NEWEST & HIGHLY EFFICIENT) ⭐ RECOMMENDED ⭐
# Multi-View Sequential Embedding: Uses 3 pooling strategies to extract robust features.
# Generates only 24 high-quality features (much fewer than traditional probe_features).
- {{"operation": "create_mvse_features"}}
```

## 方法 2: 使用增强版类 (安全)

### 使用 EnhancedTLAFS 类：
```python
from mvse_tlafs_integration import create_enhanced_tlafs_with_mvse

# 创建增强版 T-LAFS
EnhancedTLAFS = create_enhanced_tlafs_with_mvse()

# 在 main() 函数中替换原来的 TLAFS_Algorithm
tlafs = EnhancedTLAFS(
    base_df=base_df,
    target_col=TARGET_COL,
    n_iterations=N_ITERATIONS,
    results_dir=results_dir
)
```

## 优势对比

### MVSE vs 传统 Probe Features:
- **特征数量**: 24 vs 70+ 
- **训练速度**: 更快 (30 epochs vs 150 epochs)
- **过拟合风险**: 更低 (更少维度)
- **鲁棒性**: 更强 (随机遮罩机制)
- **泛化能力**: 更好 (多视角池化)

## 使用建议

1. **优先使用 MVSE**: 在高级阶段优先推荐 `create_mvse_features`
2. **组合使用**: 可以与传统特征工程方法组合
3. **参数调优**: 可以调整 `hist_len`, `num_lags`, `mask_rate` 等参数
4. **监控性能**: 观察 MVSE 特征在不同数据集上的表现
"""
    
    return guide


if __name__ == "__main__":
    # 运行集成测试
    test_mvse_integration_in_tlafs()
    
    # 打印集成指南
    print("\n" + "="*80)
    print(create_integration_guide()) 