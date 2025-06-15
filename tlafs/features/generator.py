import pandas as pd
import numpy as np
import torch

# 假设相关的模型和函数可以从框架的其他部分导入
# from ..models.autoencoder import MaskedEncoder
# from ..models.probe import ProbeForecaster
# from .mvse import generate_mvse_features_for_tlafs

# 在实际应用中，下面的TLAFS_Algorithm属性需要被正确设置或传入
# TLAFS_Algorithm.pretrained_encoders = {}
# TLAFS_Algorithm.embedder_scalers = {}
# TLAFS_Algorithm.meta_forecast_models = {}
# TLAFS_Algorithm.meta_scalers = {}
# TLAFS_Algorithm.probe_config = {}
# TLAFS_Algorithm.probe_model_path = ""
# TLAFS_Algorithm.pretrain_cols_static = []
# TLAFS_Algorithm.target_col_static = ""

def execute_plan(df: pd.DataFrame, plan: list, tlafs_params: dict):
    """
    在一个给定的DataFrame上执行特征工程计划。
    这是所有特征生成的单一、权威的静态方法。
    它包含了所有无泄漏的特征生成逻辑。
    """
    temp_df = df.copy()
    new_feature_name = None # 用于跟踪新生成的特征

    # 从传入的字典中获取必要的参数
    target_col = tlafs_params.get("target_col_static")
    pretrained_encoders = tlafs_params.get("pretrained_encoders", {})
    embedder_scalers = tlafs_params.get("embedder_scalers", {})
    meta_forecast_models = tlafs_params.get("meta_forecast_models", {})
    meta_scalers = tlafs_params.get("meta_scalers", {})
    # ... 其他需要的参数 ...

    required_time_cols = ['dayofweek', 'month', 'weekofyear', 'is_weekend']
    if not all(col in temp_df.columns for col in required_time_cols):
        # ... (确保基础时间特征存在的逻辑) ...
        pass
    
    # --- 核心修改：确保plan是一个列表 ---
    if not isinstance(plan, list):
        plan = [plan] # 如果plan是单个字典，将其包装在列表中

    executed_feature_names = []

    for step in plan:
        op = None
        try:
            op = step.get("function")
            args = step.get("args", {})
            
            # 用于记录当前步骤生成特征的临时变量
            step_feature_name = None

            # --- 新增的宏功能 ---
            if op == "create_control_baseline_features":
                print("  - 正在执行宏操作: create_control_baseline_features...")
                target = args.get("col", target_col)
                
                # 1. 时间特征 (与control.py一致)
                temp_df['year'] = pd.to_datetime(temp_df['date']).dt.year
                temp_df['month'] = pd.to_datetime(temp_df['date']).dt.month
                temp_df['day'] = pd.to_datetime(temp_df['date']).dt.day
                temp_df['dayofweek'] = pd.to_datetime(temp_df['date']).dt.dayofweek
                
                # 2. 滞后特征
                for lag in [1, 2, 3, 7, 14]:
                    temp_df[f'{target}_lag_{lag}'] = temp_df[target].shift(lag)
                
                # 3. 滚动统计特征
                for window in [7, 14, 30]:
                    temp_df[f'{target}_rolling_mean_{window}'] = temp_df[target].rolling(window=window).mean().shift(1)
                    temp_df[f'{target}_rolling_std_{window}'] = temp_df[target].rolling(window=window).std().shift(1)
                    temp_df[f'{target}_rolling_min_{window}'] = temp_df[target].rolling(window=window).min().shift(1)
                    temp_df[f'{target}_rolling_max_{window}'] = temp_df[target].rolling(window=window).max().shift(1)
                
                # 由于创建了大量滞后和滚动特征，会引入NaN，这里我们用0填充
                # 这与control.py中的dropna()行为不同，但更符合T-LAFS的迭代性质
                temp_df.fillna(0, inplace=True)
                
                # 对于宏操作，我们返回一个描述性的名字
                step_feature_name = "control_baseline_features_set"
                print("  - ✅ 成功生成了control基线特征集。")

            # --- 基础时序特征 ---
            elif op == "create_lag_features":
                col = args.get("col")
                lags = args.get("lags", [1])
                for lag in lags:
                    step_feature_name = f"{col}_lag_{lag}"
                    temp_df[step_feature_name] = temp_df[col].shift(lag).ffill().fillna(0)

            elif op == "create_rolling_features":
                col = args.get("col")
                windows = args.get("windows", [7])
                aggs = args.get("aggs", ['mean'])
                for window in windows:
                    for agg in aggs:
                        step_feature_name = f"{col}_rolling_{agg}_{window}"
                        temp_df[step_feature_name] = temp_df[col].rolling(window=window).agg(agg).shift(1).ffill().fillna(0)

            elif op == "create_interaction_features":
                col1 = args.get("col1")
                col2 = args.get("col2")
                # --- 数据泄露防火墙 ---
                if col1 == target_col or col2 == target_col:
                    print(f"  - 🛑 数据泄露警告: 交互特征不能直接使用原始目标列 ('{target_col}')。跳过此步骤。")
                    continue
                
                if col1 in temp_df.columns and col2 in temp_df.columns:
                    step_feature_name = f"{col1}_x_{col2}"
                    temp_df[step_feature_name] = temp_df[col1] * temp_df[col2]
                else:
                    print(f"  - ⚠️ 交互特征的列不存在: {col1} or {col2}。跳过此步骤。")
                    continue
            
            elif op == "create_fourier_features":
                col = args.get("col")
                order = int(args.get("order", 1))
                if col != 'date' or col not in temp_df.columns:
                    print(f"  - ⚠️ 傅里叶特征必须基于'date'列。跳过此步骤。")
                    continue
                
                print(f"  - 正在为 '{col}' 创建 {order} 阶傅里叶特征...")
                day_of_year = pd.to_datetime(temp_df[col]).dt.dayofyear
                year_length = 365.25
                
                for k in range(1, order + 1):
                    sin_col = f"fourier_sin_{k}"
                    cos_col = f"fourier_cos_{k}"
                    temp_df[sin_col] = np.sin(2 * np.pi * k * day_of_year / year_length)
                    temp_df[cos_col] = np.cos(2 * np.pi * k * day_of_year / year_length)
                
                # 对于多特征操作，返回一个描述性名称
                step_feature_name = f"fourier_features_order_{order}"

            elif op == "delete_features":
                cols_to_delete = args.get("cols", [])
                if not isinstance(cols_to_delete, list):
                    cols_to_delete = [cols_to_delete]
                
                existing_cols = [c for c in cols_to_delete if c in temp_df.columns]
                if existing_cols:
                    print(f"  - 正在删除特征: {existing_cols}")
                    temp_df.drop(columns=existing_cols, inplace=True)
                    step_feature_name = f"deleted_{len(existing_cols)}_features"
                else:
                    print(f"  - ⚠️ 想要删除的特征不存在: {cols_to_delete}。跳过此步骤。")
                    continue

            # --- 从 specialist_tlafs_experiment.py 中添加所有其他特征生成逻辑 ---
            # ... 例如: create_diff, create_ewm, 
            # ... create_time_features, create_fourier_features,
            # ... create_embedding_features, create_forecast_feature, etc.
            
            elif op == "create_embedding_features": # 同样更新为新的函数/参数格式
                col = args.get("col") # 虽然未使用，但保持一致性
                window_size = args.get("window_size", 90)
                
                embedder = pretrained_encoders.get(window_size)
                scaler = embedder_scalers.get(window_size)

                if embedder and scaler:
                    pretrain_cols = tlafs_params.get("pretrain_cols_static", [])
                    
                    if not all(c in temp_df.columns for c in pretrain_cols):
                         temp_df['dayofweek'] = temp_df['date'].dt.dayofweek
                         temp_df['month'] = temp_df['date'].dt.month
                         temp_df['weekofyear'] = temp_df['date'].dt.isocalendar().week.astype(int)
                         temp_df['is_weekend'] = (temp_df['date'].dt.dayofweek >= 5).astype(int)

                    print(f"  - 正在从 {len(pretrain_cols)} 个特征生成多变量嵌入 (窗口:{window_size})...")
                    df_for_embedding = temp_df[pretrain_cols]
                    scaled_features = scaler.transform(df_for_embedding)
                    
                    sequences = np.array([scaled_features[i:i+window_size] for i in range(len(scaled_features) - window_size + 1)])
                    
                    if sequences.size == 0:
                        print(f"  - ⚠️ 数据不足以创建窗口为 {window_size} 的嵌入。跳过。")
                        continue
                        
                    tensor = torch.FloatTensor(sequences)
                    with torch.no_grad():
                        embeddings = embedder(tensor).numpy()
                        
                    valid_indices = temp_df.index[window_size-1:]
                    # 由于可能一次性生成多个特征，我们只将第一个作为"新特征"返回以供记录
                    cols = [f"embed_{i}_win{window_size}" for i in range(embeddings.shape[1])]
                    step_feature_name = cols[0]
                    
                    embed_df = pd.DataFrame(embeddings, index=valid_indices, columns=cols)
                    
                    existing_cols_to_drop = [c for c in cols if c in temp_df.columns]
                    if existing_cols_to_drop:
                        temp_df.drop(columns=existing_cols_to_drop, inplace=True)

                    temp_df = temp_df.join(embed_df)
                    temp_df[cols] = temp_df[cols].shift(1).ffill().fillna(0)
                else:
                    print(f"  - ⚠️ 窗口 {window_size} 的嵌入器不可用。跳过此步骤。")
                    continue

            # --- 其他操作的elif块 ---
            else:
                print(f"  - ⚠️ 未知的操作: {op}。跳过此步骤。")
                continue
            
            # 如果步骤成功，记录其名称
            if step_feature_name:
                executed_feature_names.append(step_feature_name)

        except Exception as e:
            import traceback
            print(f"  - ❌ 执行步骤 {step} 时发生严重错误，已跳过。错误: {e}\n{traceback.format_exc()}")
            continue
    
    # 最终返回逻辑
    if not df.equals(temp_df) and executed_feature_names:
        # 如果Dataframe有变化，并且我们成功执行了至少一个步骤
        final_name = ", ".join(executed_feature_names)
        return temp_df, final_name
    else:
        # 如果计划为空，或所有步骤都失败/跳过
        print("  - ⚠️ 计划执行后未成功生成任何新特征。")
        return None, None

    return temp_df 