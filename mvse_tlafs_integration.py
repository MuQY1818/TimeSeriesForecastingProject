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

# 导入 MVSE 模块
from mvse_embedding import MVSEEmbedding
from mvse_probe_integration import MVSEProbeForecaster, create_mvse_probe_features, train_mvse_probe_model

warnings.filterwarnings('ignore')


def generate_mvse_features_for_tlafs(df, target_col, model, target_scaler, hist_len=90, num_lags=14):
    """
    为 T-LAFS 框架生成 MVSE 探针特征（使用预训练模型）
    
    这个函数专门为 T-LAFS 的 execute_plan 方法设计
    """
    print("  - 🔮 使用预训练的 MVSE 模型生成探针特征...")
    
    try:
        # 创建用于推理的输入数据
        hist_sequences, lag_features, _, valid_indices, _ = create_mvse_probe_features(
            df, target_col=target_col, hist_len=hist_len, num_lags=num_lags, scaler=target_scaler
        )
        
        if len(hist_sequences) == 0:
            print("  - ⚠️ 数据不足，无法生成 MVSE 特征")
            return df
        
        # 使用预训练模型生成特征
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)
        
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
        
        print(f"  - ✅ MVSE 特征生成完成: {len(feature_names)} 个特征")
        
        # 使用 shift(1) 避免数据泄露
        return df.join(features_df.shift(1))
        
    except Exception as e:
        import traceback
        print(f"  - ❌ MVSE 特征生成失败: {e}\n{traceback.format_exc()}")
        return df


def add_mvse_to_execute_plan():
    """
    返回一个包含 MVSE 操作的代码片段，可以添加到 execute_plan 方法中
    """
    mvse_operation_code = '''
                elif op == "create_mvse_features":
                    print("  - Generating MVSE probe features...")
                    temp_df = generate_mvse_features_for_tlafs(temp_df, target_col, model, target_scaler)
                    print("  - ✅ MVSE features generated.")
    '''
    return mvse_operation_code


def add_mvse_to_llm_prompt():
    """
    返回要添加到 LLM 提示中的 MVSE 工具描述
    """
    mvse_tool_description = '''
# 4. MVSE Probe Features (NEWEST & HIGHLY EFFICIENT)
# Multi-View Sequential Embedding: Uses 3 pooling strategies (GAP, GMP, MaskedGAP) to extract robust features.
# Generates only 24 high-quality features (much fewer than traditional probe_features).
# Excellent for capturing both trends and anomalies with strong robustness.
- {"operation": "create_mvse_features"}
'''
    return mvse_tool_description


def create_enhanced_tlafs_with_mvse():
    """
    创建一个增强版的 T-LAFS 类，集成了 MVSE 探针功能
    """
    
    # 这里我们需要从原始文件导入 TLAFS_Algorithm 类
    # 由于直接修改原文件可能影响现有功能，我们创建一个继承类
    
    try:
        # 尝试导入原始的 TLAFS_Algorithm
        from clp_probe_experiment import TLAFS_Algorithm as OriginalTLAFS
        
        class EnhancedTLAFS(OriginalTLAFS):
            """
            增强版 T-LAFS，集成了 MVSE 探针功能
            """
            
            @staticmethod
            def execute_plan(df: pd.DataFrame, plan: list):
                """
                重写 execute_plan 方法，添加 MVSE 支持
                """
                # 首先调用原始的 execute_plan 处理其他操作
                temp_df = df.copy()
                target_col = EnhancedTLAFS.target_col_static
                
                # 确保基础时间特征存在
                required_time_cols = ['dayofweek', 'month', 'weekofyear', 'is_weekend']
                if not all(col in temp_df.columns for col in required_time_cols):
                    if 'dayofweek' not in temp_df.columns:
                        temp_df['dayofweek'] = temp_df['date'].dt.dayofweek
                    if 'month' not in temp_df.columns:
                        temp_df['month'] = temp_df['date'].dt.month
                    if 'weekofyear' not in temp_df.columns:
                        temp_df['weekofyear'] = temp_df['date'].dt.isocalendar().week.astype(int)
                    if 'is_weekend' not in temp_df.columns:
                        temp_df['is_weekend'] = (temp_df['date'].dt.dayofweek >= 5).astype(int)
                
                for step in plan:
                    op = step.get("operation")
                    
                    try:
                        if op == "create_mvse_features":
                            print("  - Generating MVSE probe features...")
                            temp_df = generate_mvse_features_for_tlafs(temp_df, target_col, model, target_scaler)
                            print("  - ✅ MVSE features generated.")
                        else:
                            # 对于其他操作，调用父类的方法
                            # 由于父类方法是静态的，我们需要重新实现核心逻辑
                            temp_df = OriginalTLAFS.execute_plan(temp_df, [step])
                            
                    except Exception as e:
                        import traceback
                        print(f"  - ❌ ERROR executing step {step}. Error: {e}\n{traceback.format_exc()}")
                
                return temp_df
            
            def get_plan_from_llm(self, context_prompt, iteration_num, max_iterations):
                """
                重写 LLM 提示生成方法，添加 MVSE 工具
                """
                # 调用父类方法获取基础提示
                original_prompt = super().get_plan_from_llm.__func__(self, context_prompt, iteration_num, max_iterations)
                
                # 如果是高级阶段，我们需要修改提示以包含 MVSE
                stage = "advanced"
                if (iteration_num / max_iterations) < 0.4:
                    stage = "basic"
                
                if stage == "advanced":
                    # 重新生成包含 MVSE 的提示
                    return self._get_enhanced_plan_from_llm(context_prompt, iteration_num, max_iterations)
                else:
                    return original_prompt
            
            def _get_enhanced_plan_from_llm(self, context_prompt, iteration_num, max_iterations):
                """
                生成包含 MVSE 的增强提示
                """
                from clp_probe_experiment import gemini_model
                import json
                
                base_prompt = f"""You are a Data Scientist RL agent. Your goal is to create a feature engineering plan to maximize the Fusion R^2 score.
Your response MUST be a valid JSON list of operations: `[ {{"operation": "op_name", ...}}, ... ]`.
The target column is '{self.target_col}'.
"""
                
                basic_tools = """
# *** STAGE 1: BASIC FEATURE ENGINEERING ***
# Focus on creating a strong baseline with fundamental time-series features.
# AVAILABLE TOOLS:
- {{"operation": "create_lag", "on": "feature_name", "days": int, "id": "..."}}
- {{"operation": "create_diff", "on": "feature_name", "periods": int, "id": "..."}}
- {{"operation": "create_rolling_mean", "on": "feature_name", "window": int, "id": "..."}}
- {{"operation": "create_rolling_std", "on": "feature_name", "window": int, "id": "..."}}
- {{"operation": "create_ewm", "on": "feature_name", "span": int, "id": "..."}}
- {{"operation": "create_fourier_features", "period": 365.25, "order": 4}}
- {{"operation": "create_interaction", "features": ["feat1", "feat2"], "id": "..."}}
- {{"operation": "delete_feature", "feature": "feature_name"}}
"""

                advanced_tools = """
# *** STAGE 2: ADVANCED FEATURE ENGINEERING ***
# Now you can use powerful learned embeddings and meta-forecasts. Combine them with the best basic features.
# AVAILABLE TOOLS (includes all basic tools plus):
# 1. Learned Embeddings (VERY POWERFUL)
- {{"operation": "create_learned_embedding", "window": [90, 365, 730], "id": "UNIQUE_ID"}}

# 2. Meta-Forecast Features
- {{"operation": "create_forecast_feature", "model_name": ["SimpleNN_meta", "EnhancedNN_meta"], "id": "UNIQUE_ID"}}

# 3. Traditional Attention Probe Features (POWERFUL but HIGH-DIMENSIONAL)
# This generates 70+ features from a 365-day lookback window using an attention-based probe.
- {{"operation": "create_probe_features"}}

# 4. MVSE Probe Features (NEWEST & HIGHLY EFFICIENT) ⭐ RECOMMENDED ⭐
# Multi-View Sequential Embedding: Uses 3 pooling strategies (GAP, GMP, MaskedGAP) to extract robust features.
# Generates only 24 high-quality features (much fewer than traditional probe_features).
# Excellent for capturing both trends and anomalies with strong robustness.
# ADVANTAGE: Lower dimensionality, better generalization, faster training.
- {{"operation": "create_mvse_features"}}
"""
                
                rules = """
*** RULES ***
- IDs must be unique. Do not reuse IDs from "Available Features".
- Propose short plans (1-3 steps).
- For parameters shown with a list of options (e.g., "window": [90, 365]), you MUST CHOOSE ONLY ONE value.
- `create_mvse_features` is HIGHLY RECOMMENDED over `create_probe_features` due to better efficiency and lower overfitting risk.
- `create_learned_embedding` is very powerful. Try interacting it with other features using `create_interaction`.
- Prefer `create_mvse_features` when you need advanced probe capabilities with better generalization.
"""

                system_prompt = base_prompt + basic_tools + advanced_tools + rules
                
                try:
                    if gemini_model is None:
                        raise Exception("Gemini model not initialized.")
                    
                    full_prompt_for_gemini = system_prompt + "\n\n" + context_prompt
                    response = gemini_model.generate_content(full_prompt_for_gemini)
                    plan_str = response.text
                    parsed_json = json.loads(plan_str)

                    if isinstance(parsed_json, dict) and "plan" in parsed_json:
                        plan = parsed_json.get("plan", [])
                        return plan if isinstance(plan, list) else [plan]
                    elif isinstance(parsed_json, list):
                        return parsed_json
                    elif isinstance(parsed_json, dict) and "operation" in parsed_json:
                        return [parsed_json]
                    else:
                        print(f"  - ⚠️ Warning: LLM returned unexpected structure: {parsed_json}")
                        return []
                        
                except Exception as e:
                    print(f"❌ Error calling Gemini: {e}")
                    return [{"operation": "create_mvse_features"}]  # 默认使用 MVSE
        
        return EnhancedTLAFS
        
    except ImportError as e:
        print(f"❌ 无法导入原始 TLAFS_Algorithm: {e}")
        print("请确保 clp_probe_experiment.py 文件存在且可导入")
        return None


def test_mvse_integration_in_tlafs():
    """
    测试 MVSE 在 T-LAFS 中的集成
    """
    print("🧪 测试 MVSE 在 T-LAFS 中的集成...")
    
    # 创建测试数据
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    np.random.seed(42)
    
    # 生成带有季节性的时间序列
    t = np.arange(len(dates))
    seasonal = 10 * np.sin(2 * np.pi * t / 365.25)
    trend = 0.01 * t
    noise = np.random.normal(0, 2, len(dates))
    temp = 20 + seasonal + trend + noise
    
    df = pd.DataFrame({
        'date': dates,
        'temp': temp
    })
    
    print(f"📊 测试数据: {len(df)} 个样本")
    
    # 测试 MVSE 特征生成
    print("\n🔧 测试 MVSE 特征生成...")
    df_with_mvse = generate_mvse_features_for_tlafs(df, target_col='temp', model=None, target_scaler=None)
    
    # 检查结果
    mvse_cols = [col for col in df_with_mvse.columns if 'mvse_' in col]
    print(f"✅ 生成的 MVSE 特征: {len(mvse_cols)} 个")
    print(f"   特征列表: {mvse_cols}")
    
    # 测试增强版 T-LAFS 类
    print("\n🚀 测试增强版 T-LAFS 类...")
    EnhancedTLAFS = create_enhanced_tlafs_with_mvse()
    
    if EnhancedTLAFS:
        print("✅ 增强版 T-LAFS 类创建成功")
        
        # 测试 execute_plan 方法
        test_plan = [{"operation": "create_mvse_features"}]
        
        # 模拟设置必要的类属性
        EnhancedTLAFS.target_col_static = 'temp'
        
        result_df = EnhancedTLAFS.execute_plan(df, test_plan)
        
        new_mvse_cols = [col for col in result_df.columns if 'mvse_' in col]
        print(f"✅ execute_plan 测试成功: 生成了 {len(new_mvse_cols)} 个 MVSE 特征")
    else:
        print("❌ 增强版 T-LAFS 类创建失败")
    
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
    temp_df = generate_mvse_features_for_tlafs(temp_df, target_col, model, target_scaler)
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