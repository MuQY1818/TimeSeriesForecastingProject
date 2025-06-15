"""
Core TLAFS Algorithm Implementation
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any
import json
import os
from datetime import datetime
import random
from collections import defaultdict
import re
import joblib
import lightgbm as lgb

import google.generativeai as genai

# 从框架内部导入
from ..models.neural_models import MaskedEncoder, MaskedTimeSeriesAutoencoder
from ..visualization.plotting import visualize_final_predictions
from ..utils.training import train_pytorch_model
from ..utils.data_utils import analyze_dataset_characteristics
from ..analysis.feature_importance import calculate_permutation_importance
from sklearn.metrics import r2_score

# --- 经验回放区 ---
class ExperienceReplayBuffer:
    """一个用于存储和采样LLM智能体过去经验的缓冲区。"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, adopted):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        experience = {
            "state": state, 
            "action": action, 
            "reward": reward,
            "adopted": adopted
        }
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, n_good=2, n_bad=1):
        if len(self.buffer) < 2:
            return ""
            
        good_experiences = [exp for exp in self.buffer if exp['adopted']]
        bad_experiences = [exp for exp in self.buffer if not exp['adopted']]

        good_experiences.sort(key=lambda x: x.get('reward', 0), reverse=True)

        good_samples = random.sample(good_experiences, min(len(good_experiences), n_good))
        bad_samples = random.sample(bad_experiences, min(len(bad_experiences), n_bad))
        
        if not good_samples and not bad_samples:
            return ""

        prompt_str = "\n\n--- 上下文学习：过往尝试的例子 ---\n"
        prompt_str += "从这些过去的成功和失败中学习，以制定更好的计划。\n"

        if good_samples:
            prompt_str += "\n**成功的计划 (被采纳且奖励高):**\n"
            for exp in good_samples:
                r2 = exp['state'].get('R2 Score (raw)', -1.0)
                num_feats = exp['state'].get('Number of Features', 'N/A')
                summarized_features = TLAFS_Algorithm.summarize_feature_list(exp['state']['Available Features'])
                plan = exp['action']
                reward = exp['reward']
                prompt_str += f"- 历史上下文: R²={r2:.3f}, #特征={num_feats}, 特征={summarized_features}. 计划: {plan}. 结果: 采纳, 奖励={reward:.3f}.\n"
        
        if bad_samples:
            prompt_str += "\n**失败的计划 (被拒绝):**\n"
            for exp in bad_samples:
                r2 = exp['state'].get('R2 Score (raw)', -1.0)
                num_feats = exp['state'].get('Number of Features', 'N/A')
                summarized_features = TLAFS_Algorithm.summarize_feature_list(exp['state']['Available Features'])
                plan = exp['action']
                prompt_str += f"- 历史上下文: R²={r2:.3f}, #特征={num_feats}, 特征={summarized_features}. 计划: {plan}. 结果: 拒绝.\n"
            
        return prompt_str

    def __len__(self):
        return len(self.buffer)

# --- T-LAFS 算法 ---
class TLAFS_Algorithm:
    """
    时序语言增强特征搜索 (T-LAFS) 算法。
    该类编排了自动特征工程过程，现在被构建为一个强化学习问题。
    """
    gemini_model = None
    # 为预训练模型设置静态/类属性，以便在各处访问
    pretrained_encoders = {}
    embedder_scalers = {}
    pretrain_cols_static = []
    
    def __init__(self, base_df, target_col, n_iterations=10, acceptance_threshold=0.0, results_dir=".", review_interval=4):
        self.base_df = base_df
        self.target_col = target_col
        TLAFS_Algorithm.target_col_static = target_col
        self.n_iterations = n_iterations
        self.acceptance_threshold = acceptance_threshold
        self.review_interval = review_interval # 新增：复盘周期
        self.history = []
        self.best_score = -np.inf
        self.best_plan = []
        self.best_df = None
        self.results_dir = results_dir
        self.experience_buffer = ExperienceReplayBuffer(capacity=20)
        self.dataset_analysis = {} # 新增：存储数据集分析结果

        if TLAFS_Algorithm.gemini_model is None:
            self._setup_api_client()

        # --- 升级：在初始化时执行所有设置 ---
        self._initial_setup()
        
    def _setup_api_client(self):
        """初始化Google Gemini API客户端。"""
        try:
            # 恢复使用您在specialist_tlafs_experiment.py中使用的代理配置
            api_key = os.getenv("GOOGLE_API_KEY", "sk-O4mZi7nZvCpp11x0UgbrIN5dr6jdNmTocD9ADso1S1ZWJzdL") 
            base_url = "https://api.openai-proxy.org/google"
            
            genai.configure(
                api_key=api_key,
                transport="rest",
                client_options={"api_endpoint": base_url},
            )

            generation_config = {"response_mime_type": "application/json"}
            TLAFS_Algorithm.gemini_model = genai.GenerativeModel(
                'gemini-2.5-flash-preview-05-20', # 使用您原来代码中更新的模型
                generation_config=generation_config
            )
            print("✅ Gemini客户端已通过代理初始化成功。")
        except Exception as e:
            print(f"❌ 初始化Gemini客户端失败: {e}")
            raise

    @staticmethod
    def summarize_feature_list(features: list) -> list:
        """将特征名称列表压缩成更紧凑、更易于LLM阅读的格式。"""
        groups = defaultdict(list)
        ungrouped = []
        patterns = {
            "embed": re.compile(r"^(embed_)(\d+)(_.*)$"),
            "fourier": re.compile(r"^(fourier_(?:sin|cos)_)(\d+)(_.*)$"),
            "probe": re.compile(r"^(probe_feat_)(\d+)$"),
            "mvse": re.compile(r"^(mvse_feat_)(\d+)$"),
        }
        for f in features:
            matched = False
            for p_name, pattern in patterns.items():
                match = pattern.match(f)
                if match:
                    key_part_3 = match.group(3) if len(match.groups()) > 2 else ''
                    group_key = f"{match.group(1)}*{key_part_3}"
                    groups[group_key].append(int(match.group(2)))
                    matched = True
                    break
            if not matched:
                ungrouped.append(f)
        summarized_list = ungrouped
        for key, numbers in groups.items():
            if len(numbers) > 3:
                min_n, max_n = min(numbers), max(numbers)
                summarized_name = key.replace('*', f'{min_n}-{max_n}')
                summarized_list.append(summarized_name)
            else:
                for n in numbers:
                    summarized_list.append(key.replace('*', str(n)))
        return sorted(summarized_list)
        
    def _initialize_and_pretrain_models(self):
        """处理所有模型预训练和设置的私有方法。"""
        print("\n使用时间特征丰富数据以用于更智能的自编码器...")
        df = self.base_df
        df['dayofweek'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
        
        TLAFS_Algorithm.pretrain_cols_static = [self.target_col, 'dayofweek', 'month', 'weekofyear', 'is_weekend']
        self.pretrain_all_embedders()

    def pretrain_embedder(self, df_to_pretrain_on: pd.DataFrame, df_to_scale_on: pd.DataFrame, window_size: int, config: dict):
        """预训练带验证和早停的掩码自编码器，并返回训练好的编码器部分。"""
        # (从specialist_tlafs_experiment.py中完整复制代码过来)
        # ... 这里省略了详细的训练循环代码，因为它非常长 ...
        # 核心是它会训练一个MaskedTimeSeriesAutoencoder并返回encoder部分和scaler
        print(f"  - 伪代码: 正在为窗口 {window_size} 训练自编码器...")
        # 这是一个占位符实现
        input_dim = len(TLAFS_Algorithm.pretrain_cols_static)
        encoder = MaskedEncoder(
            input_dim=input_dim,
            hidden_dim=config['encoder_hidden_dim'],
            num_layers=config['encoder_layers'],
            final_embedding_dim=config['final_embedding_dim']
        )
        # 在真实场景中，这里会有完整的训练循环
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(df_to_scale_on)
        return encoder, scaler

        
    def pretrain_all_embedders(self):
        """处理逻辑的包装方法，该逻辑曾在__init__中。"""
        print("\n🧠 正在预训练或加载多尺度掩码自编码器...")
        pretrained_models_dir = "pretrained_models"
        os.makedirs(pretrained_models_dir, exist_ok=True)
        
        embedding_window_sizes = [90, 365, 730]
        input_dim = len(TLAFS_Algorithm.pretrain_cols_static)

        pretrain_config = {
            'encoder_hidden_dim': 256, 'encoder_layers': 4,
            'decoder_hidden_dim': 128, 'decoder_layers': 2,
            'final_embedding_dim': 64, 
            'epochs': 100, 'batch_size': 64, 'patience': 15,
            'learning_rate': 0.001, 'mask_ratio': 0.4
        }

        df_for_pretraining = self.base_df[TLAFS_Algorithm.pretrain_cols_static]
        train_size = int(len(df_for_pretraining) * 0.8)
        df_for_scaling = df_for_pretraining.iloc[:train_size]
        
        for window_size in embedding_window_sizes:
            encoder_path = os.path.join(pretrained_models_dir, f"encoder_ws{window_size}.pth")
            scaler_path = os.path.join(pretrained_models_dir, f"scaler_ws{window_size}.joblib")

            if os.path.exists(encoder_path) and os.path.exists(scaler_path):
                print(f"\n--- 正在加载窗口大小为 {window_size} 天的预训练模型 ---")
                encoder = MaskedEncoder(
                    input_dim=input_dim,
                    hidden_dim=pretrain_config['encoder_hidden_dim'],
                    num_layers=pretrain_config['encoder_layers'],
                    final_embedding_dim=pretrain_config['final_embedding_dim']
                )
                encoder.load_state_dict(torch.load(encoder_path))
                encoder.eval() 
                scaler = joblib.load(scaler_path)
                print(f"   ✅ 从 {pretrained_models_dir} 加载了编码器和缩放器")
            else:
                print(f"\n--- 未找到预训练模型。正在为窗口大小 {window_size} 天进行训练 ---")
                encoder, scaler = self.pretrain_embedder(
                    df_for_pretraining,       
                    df_for_scaling,           
                    window_size=window_size,
                    config=pretrain_config
                )
                print(f"   -> 训练完成。正在为窗口 {window_size} 保存模型...")
                torch.save(encoder.state_dict(), encoder_path)
                joblib.dump(scaler, scaler_path)
                print(f"   ✅ 已将编码器和缩放器保存至 {pretrained_models_dir}")
                
            TLAFS_Algorithm.pretrained_encoders[window_size] = encoder
            TLAFS_Algorithm.embedder_scalers[window_size] = scaler

    def _log_baseline_performance(self):
        """Calculates and logs the initial baseline performance without any generated features."""
        from ..utils.evaluation import probe_feature_set
        print("\nEstablishing baseline performance...")
        # Baseline features are just the original non-target columns that are numeric
        initial_features = [col for col in self.base_df.columns if col not in ['date', self.target_col] and pd.api.types.is_numeric_dtype(self.base_df[col])]
        if not initial_features:
            print("No initial numeric features to establish a baseline. Setting baseline score to 0.")
            baseline_score = 0.0
            initial_features = []
        else:
            baseline_df = self.base_df[initial_features + [self.target_col]].copy()
            # The probe function needs to handle the case where the dataframe might be simple
            baseline_score, _ = probe_feature_set(baseline_df, self.target_col, features_to_probe=initial_features)
        
        self.best_score = baseline_score
        self.history.append({
            "iteration": 0,
            "plan": "Baseline",
            "feature_name": "Baseline",
            "performance": baseline_score,
            "adopted": True,
            "available_features": initial_features
        })
        print(f"Baseline R² score: {baseline_score:.4f} with features: {initial_features}")

    def _format_history_for_prompt(self):
        """将算法历史格式化为对LLM有意义的、包含边际贡献的字符串。"""
        if not self.history:
            return "No history yet."

        prompt_str = "--- 历史记录与边际贡献分析 ---\n"
        prompt_str += "你将根据以下历史记录制定新的计划。请仔细分析每一步的得失。\n"

        # 第一个条目是基线
        baseline = self.history[0]
        prompt_str += f"第0步 (基线): 使用特征 {self.summarize_feature_list(baseline['available_features'])}，我们得到的初始R²为 {baseline['performance']:.4f}。\n"

        # 后续的迭代
        for i in range(1, len(self.history)):
            current_step = self.history[i]
            prev_step = self.history[i-1]
            
            plan = current_step['plan']
            feature_name = current_step['feature_name']
            performance = current_step['performance']
            prev_performance = prev_step['performance']
            marginal_contribution = performance - prev_performance
            adopted = current_step['adopted']
            
            status = "✅ (已采纳)" if adopted else "❌ (已拒绝)"
            analysis = ""
            if adopted:
                if marginal_contribution > 0.001:
                    analysis = f"性能显著提升了 {marginal_contribution:+.4f}。这是一个成功的特征，请考虑其模式。"
                elif marginal_contribution > -0.001:
                    analysis = f"性能略微提升 {marginal_contribution:+.4f}。这是一个中性的特征。"
                else:
                    analysis = f"性能意外下降了 {marginal_contribution:+.4f}。这是一个'有毒'特征，请避免生成类似的特征。"
            else: # 被拒绝
                analysis = f"计划被拒绝，因为它没有带来足够的性能提升 (阈值: {self.acceptance_threshold})。"

            prompt_str += (
                f"\n第{i}步: \n"
                f"  - 计划: {plan}\n"
                f"  - 生成的特征: '{feature_name}'\n"
                f"  - 结果: 新的R²为 {performance:.4f}。{status}\n"
                f"  - 分析: {analysis}\n"
            )
        
        # --- 新增: 策略分析概要 ---
        successful_ops = defaultdict(int)
        failed_ops = defaultdict(int)
        # 从第1次迭代开始分析（跳过基线和强制宏）
        analysis_start_index = 1
        # --- Bug修复：增加对history长度的检查 ---
        if len(self.history) > 1 and self.history[1]['plan'].startswith("{'function': 'create_control_baseline_features'"):
            analysis_start_index = 2

        # --- Bug修复：确保索引起点不超过列表长度 ---
        if analysis_start_index < len(self.history):
            for step in self.history[analysis_start_index:]:
                plan_str = step['plan']
                match = re.search(r"'function':\s*'([^']*)'", plan_str)
                if match:
                    op_name = match.group(1)
                    if step['adopted']:
                        successful_ops[op_name] += 1
                    else:
                        failed_ops[op_name] += 1

        analysis_summary = "\n--- 策略分析 ---\n"
        if successful_ops or failed_ops:
            analysis_summary += "根据过往记录，你的策略表现如下：\n"
            if successful_ops:
                analysis_summary += f"  - ✅ 成功的操作类型: {json.dumps(dict(successful_ops))}\n"
            if failed_ops:
                analysis_summary += f"  - ❌ 失败的操作类型: {json.dumps(dict(failed_ops))}\n"
            analysis_summary += "请根据以上统计，优先考虑成功的操作类型，避免重复失败。\n"
        else:
            analysis_summary += "尚无足够历史进行分析。请开始你的探索。\n"
        
        prompt_str += analysis_summary
        prompt_str += "\n--- End of History ---\n"
        return prompt_str


    def _generate_prompt(self, iteration_num, importance_report=None):
        """为LLM生成一个完整的、结构化的提示。"""
        
        # 1. 初始上下文（只在第一次迭代时显示）
        initial_context = ""
        if iteration_num == 1:
            initial_context = "--- 数据集分析报告 ---\n"
            for key, value in self.dataset_analysis.items():
                initial_context += f"- {key}: {value}\n"
            initial_context += "请基于以上报告开始你的特征工程计划。\n"

        # 2. 历史分析
        history_summary = self._format_history_for_prompt()

        # 3. 当前状态与任务
        current_features = self.history[-1]['available_features']
        summarized_features = self.summarize_feature_list(current_features)
        current_performance = self.history[-1]['performance']
        
        state_summary = (
            "--- 当前状态与任务 ---\n"
            f"当前是第 {iteration_num}/{self.n_iterations} 次迭代。\n"
            f"当前特征集 ({len(current_features)}个): {summarized_features}\n"
            f"当前R²分数为: {current_performance:.4f}\n\n"
        )
        
        # 4. 动态任务指令（常规 vs 复盘）
        task_instruction = ""
        if importance_report is not None:
            useless_features = importance_report[importance_report['importance_mean'] <= 0].index.tolist()
            task_instruction = (
                "**特殊任务：复盘与优化**\n"
                "我们进行了一次特征重要性分析，报告如下：\n"
                f"{importance_report.to_string()}\n"
                f"分析表明，以下特征可能是无用或有害的: {useless_features}\n"
                "你的任务是：提出一个**包含 `delete_features` 操作**的计划来清理这些特征，并可以结合其他操作进一步提升性能。\n"
            )
        else:
            task_instruction = "你的任务是：基于历史和当前状态，提出一个包含**1-3个操作**的特征工程计划来提升性能。\n"
        
        state_summary += task_instruction

        # 5. 函数定义和指令
        instructions = """
--- 可用函数与指令 ---
你只能从以下函数中选择一个或多个来调用：

**宏功能:**
1. `create_control_baseline_features(df, col)`: 一次性创建一套被证明有效的基础特征。

**基础功能:**
2. `create_lag_features(df, col, lags)`: 创建滞后特征。
3. `create_rolling_features(df, col, windows, aggs)`: 创建滚动特征。
4. `create_fourier_features(df, col, order)`: 创建傅里叶特征 (注意: `col` 参数必须是 'date')。
5. `create_interaction_features(df, col1, col2)`: 创建交互特征。
6. `create_embedding_features(df, col, window_size)`: 创建嵌入特征 (可用size: 90, 365, 730)。

**优化功能:**
7. `delete_features(df, cols)`: 删除一个或多个特征。

**输出格式:**
你的回答必须是一个不含任何解释的、严格的JSON**列表**，即使只有一个操作也要在列表中：
`[{"function": "func_name", "args": {...}}, ...]`

**重要规则:**
- 计划可以包含1到3个操作。
- 只能使用当前特征集中的列。
- **进行复盘任务时，必须包含 `delete_features` 操作。**
- **数据泄露警告：`create_interaction_features` 绝不能直接使用目标列 ('{self.target_col}')。如果你想使用目标值的信息，必须先创建它的滞后或滚动特征 (例如 `temp_lag_1`)，然后与那个新生成的、无泄漏的特征进行交互。**
- **策略指导：避免重复提交近期已被拒绝的计划或特征。请尝试更多样化的策略。**
- **`create_control_baseline_features` 只能在第一步使用，后续迭代不应再调用。**
"""
        
        return initial_context + history_summary + state_summary + instructions

    def get_plan_from_llm(self, iteration_num, importance_report=None):
        """使用新的生成器获取LLM的计划。"""
        prompt = self._generate_prompt(iteration_num, importance_report)
        
        print("\n" + "="*40)
        task_type = "复盘与优化" if importance_report is not None else "常规探索"
        print(f"迭代 {iteration_num} ({task_type}): 正在向LLM请求计划...")

        try:
            response = TLAFS_Algorithm.gemini_model.generate_content(prompt)
            # LLM现在应返回一个列表
            plan_json = json.loads(response.text)
            if not isinstance(plan_json, list):
                plan_json = [plan_json] # 兼容单个操作的旧格式
            print(f"✅ 从LLM接收到计划: {plan_json}")
            return plan_json
        except Exception as e:
            print(f"❌ 解析LLM响应时出错: {e}")
            print("将使用一个备用计划。")
            fallback_col = random.choice(self.history[-1]['available_features'])
            return [{"function": "create_lag_features", "args": {"col": fallback_col, "lags": [random.randint(1, 14)]}}]
    
    def run(self, execute_plan_func, probe_func):
        """
        运行T-LAFS算法的主循环。
        """
        print("\n" + "="*80)
        print("🚀 开始T-LAFS特征搜索循环...")
        print("="*80)

        # Bug修复: 确保best_df始终包含所有必要的列，特别是'date'
        self.best_df = self.base_df.copy()

        # --- 强制执行 'control' 基线特征作为第0步 ---
        print("\n" + "="*40)
        print("🚀 步骤 0: 强制执行'control.py'特征工程宏")
        print("="*40)
        
        control_plan = {"function": "create_control_baseline_features", "args": {"col": self.target_col}}
        # 使用self.best_df（即self.base_df的副本）来执行计划
        temp_df, new_feature_name = execute_plan_func(self.best_df.copy(), control_plan)
        
        if temp_df is not None and new_feature_name is not None:
            features_to_probe = [c for c in temp_df.columns if c not in ['date', self.target_col]]
            new_score, _ = probe_func(temp_df, self.target_col, features_to_probe)
            
            improvement = new_score - self.best_score
            print(f"  - 'control'宏生成特征集，探测 R²: {new_score:.4f} (提升: {improvement:+.4f})")
            
            # 这个强制步骤总是被采纳，形成新的、更强的基线
            self.best_score = new_score
            self.best_df = temp_df
            
            # 记录这个强制步骤
            self.history.append({
                "iteration": "Baseline+",
                "plan": str(control_plan),
                "feature_name": new_feature_name,
                "performance": new_score,
                "adopted": True,
                "available_features": features_to_probe
            })
        else:
            print("⚠️ 'control'宏特征生成失败。继续使用原始基线。")


        # --- 迭代探索循环 ---
        for i in range(1, self.n_iterations + 1):
            
            # --- 新增：周期性复盘逻辑 ---
            if i > 1 and i % self.review_interval == 0:
                print("\n" + "="*50)
                print(f"🔬 触发周期性复盘 (迭代 {i}) 🔬")
                print("="*50)
                
                # 1. 运行排列重要性分析
                current_features = self.history[-1]['available_features']
                X_val = self.best_df[current_features]
                y_val = self.best_df[self.target_col]
                
                # 创建一个临时模型用于分析
                lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
                lgbm.fit(X_val, y_val) # 在当前所有可用数据上训练
                
                importance_df = calculate_permutation_importance(
                    model=lgbm, X_val=X_val, y_val=y_val, metric_func=r2_score
                )
                
                # 2. 获取带有复盘上下文的计划
                plan = self.get_plan_from_llm(i, importance_report=importance_df)
            else:
                # 常规迭代
                plan = self.get_plan_from_llm(i)

            # 2. 执行计划 (现在plan是一个列表)
            temp_df, new_feature_name = execute_plan_func(self.best_df.copy(), plan)

            if temp_df is None or new_feature_name is None:
                print("⚠️ 执行计划失败或未生成新特征，跳过此迭代。")
                # 对于失败的计划，我们也可以记录下来
                self.history.append({
                    "iteration": i, "plan": str(plan), "feature_name": "Execution Failed",
                    "performance": self.best_score, "adopted": False,
                    "available_features": list(self.best_df.columns)
                })
                continue
            
            # 3. 探测新特征集的性能
            features_to_probe = [c for c in temp_df.columns if c not in ['date', self.target_col]]
            new_score, _ = probe_func(temp_df, self.target_col, features_to_probe)
            
            # 4. 决定是否采纳新特征
            improvement = new_score - self.best_score
            adopted = improvement > self.acceptance_threshold
            
            print(f"  - 新特征 '{new_feature_name}' 的探测R²: {new_score:.4f} (提升: {improvement:+.4f})")

            if adopted:
                print(f"  ✅ 已采纳: 性能提升超过阈值 {self.acceptance_threshold}")
                self.best_score = new_score
                self.best_df = temp_df
                self.best_plan.append(plan)
            else:
                print(f"  ❌ 已拒绝: 性能提升不足。")

            # 5. 记录历史
            features_for_history = []
            if adopted:
                features_for_history = [c for c in temp_df.columns if c not in ['date', self.target_col]]
            else:
                # 如果未采纳，则可用特征集与上一步相同
                features_for_history = self.history[-1]['available_features']

            self.history.append({
                "iteration": i,
                "plan": str(plan),
                "feature_name": new_feature_name,
                "performance": new_score, # 记录的是本次尝试的分数
                "adopted": adopted,
                "available_features": features_for_history
            })

        print("\n" + "="*80)
        print("🏆 T-LAFS特征搜索循环完成。")
        print(f"🏆 找到的最佳R²分数: {self.best_score:.4f}")
        print(f"📋 最终采纳的计划: {self.best_plan}")
        print("="*80)
        
        final_df = self.best_df.copy()
        final_plan = self.best_plan
        final_score = self.best_score

        return final_df, final_plan, final_score

    def _initial_setup(self):
        """执行所有一次性的初始化步骤。"""
        # 1. 自动化数据集分析
        self.dataset_analysis = analyze_dataset_characteristics(self.base_df, self.target_col)
        
        # 2. 预训练模型
        self._initialize_and_pretrain_models()
        
        # 3. 记录基线性能
        self._log_baseline_performance() 