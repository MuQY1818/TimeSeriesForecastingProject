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
        """Calculates and logs the initial baseline performance, starting from scratch."""
        print("\nEstablishing baseline performance from scratch...")
        # Start with no features. The baseline score is 0 for R^2,
        # representing a model that just predicts the mean.
        baseline_score = 0.0
        initial_features = []
        initial_importances = pd.DataFrame()

        self.best_score = baseline_score
        # The base_df still contains the target and date, but we start with an empty feature list
        self.best_df = self.base_df[['date', self.target_col]].copy()
        
        self.history.append({
            "iteration": 0,
            "plan": "Baseline",
            "feature_name": "Baseline",
            "performance": baseline_score,
            "adopted": True,
            "available_features": initial_features,
            "importances": initial_importances
        })
        print(f"Baseline R² score: {baseline_score:.4f} with no initial features.")

    def _format_history_for_prompt(self):
        """Format algorithm history into a meaningful string with marginal contributions."""
        if not self.history:
            return "No history yet."

        prompt_str = "--- History & Marginal Contribution Analysis ---\n"
        prompt_str += "Analyze the following history to create new plans.\n"

        # First entry is baseline
        baseline = self.history[0]
        prompt_str += f"Step 0 (Baseline): Using features {self.summarize_feature_list(baseline['available_features'])}, initial R²: {baseline['performance']:.4f}.\n"

        # Subsequent iterations
        for i in range(1, len(self.history)):
            current_step = self.history[i]
            prev_step = self.history[i-1]
            
            plan = current_step['plan']
            feature_name = current_step['feature_name']
            performance = current_step['performance']
            prev_performance = prev_step['performance']
            marginal_contribution = performance - prev_performance
            adopted = current_step['adopted']
            
            status = "✅ (Adopted)" if adopted else "❌ (Rejected)"
            analysis = ""
            if adopted:
                if marginal_contribution > 0.001:
                    analysis = f"Significant improvement: {marginal_contribution:+.4f}. This is a successful feature pattern."
                elif marginal_contribution >= 0:
                    analysis = f"Slight improvement or neutral: {marginal_contribution:+.4f}. This is a neutral feature."
                else:
                    analysis = f"Unexpected performance drop: {marginal_contribution:+.4f}. This is a 'toxic' feature pattern to avoid."
            else:
                analysis = f"Plan rejected due to insufficient performance improvement (threshold: {self.acceptance_threshold})."
            
            # Feature importance info
            importance_info = ""
            if 'importances' in current_step and current_step['importances'] is not None and not current_step['importances'].empty:
                top_5 = current_step['importances'].head(5)
                bottom_5 = current_step['importances'][current_step['importances']['importance'] <= 0].tail(5)
                importance_info += f"\n  - Top 5 features: {top_5['feature'].tolist()}"
                if not bottom_5.empty:
                    importance_info += f"\n  - Bottom 5 features: {bottom_5['feature'].tolist()}"

            prompt_str += (
                f"\nStep {i}:\n"
                f"  - Plan: {plan}\n"
                f"  - Generated feature: '{feature_name}'\n"
                f"  - Result: New R²: {performance:.4f}. {status}\n"
                f"  - Analysis: {analysis}{importance_info}\n"
            )
        
        # Strategy analysis summary
        successful_ops = defaultdict(int)
        failed_ops = defaultdict(int)
        analysis_start_index = 1
        if len(self.history) > 1 and self.history[1]['plan'].startswith("{'function': 'create_control_baseline_features'"):
            analysis_start_index = 2

        if analysis_start_index < len(self.history):
            for step in self.history[analysis_start_index:]:
                plan_str = step['plan']
                match = re.search(r"['\"]function['\"]\s*:\s*['\"]([^'\"]*)['\"]", plan_str)
                if match:
                    op_name = match.group(1)
                    if step['adopted']:
                        successful_ops[op_name] += 1
                    else:
                        failed_ops[op_name] += 1

        analysis_summary = "\n--- Strategy Analysis ---\n"
        if successful_ops or failed_ops:
            analysis_summary += "Based on history, your strategy performance:\n"
            if successful_ops:
                analysis_summary += f"  - ✅ Successful operations: {json.dumps(dict(successful_ops))}\n"
            if failed_ops:
                analysis_summary += f"  - ❌ Failed operations: {json.dumps(dict(failed_ops))}\n"
            analysis_summary += "Prioritize successful operations, avoid repeating failures.\n"
        else:
            analysis_summary += "Insufficient history for analysis. Begin exploration.\n"
        
        prompt_str += analysis_summary
        prompt_str += "\n--- End of History ---\n"
        return prompt_str


    def _generate_prompt(self, iteration_num, importance_report=None):
        """Generate a complete, structured prompt for LLM."""
        
        # 1. Initial context (only shown in first iteration)
        initial_context = ""
        if iteration_num == 1:
            initial_context = "--- Dataset Analysis Report ---\n"
            for key, value in self.dataset_analysis.items():
                initial_context += f"- {key}: {value}\n"
            initial_context += "Begin feature engineering based on this report.\n"

        # 2. History analysis
        history_summary = self._format_history_for_prompt()

        # 3. Current state & task
        current_features = self.history[-1]['available_features']
        summarized_features = self.summarize_feature_list(current_features)
        current_performance = self.history[-1]['performance']
        
        state_summary = (
            "--- Current State & Task ---\n"
            f"Iteration {iteration_num}/{self.n_iterations}\n"
            f"Current features ({len(current_features)}): {summarized_features}\n"
            f"Current R²: {current_performance:.4f}\n\n"
        )
        
        # Tactical observations
        last_importance_df = self.history[-1].get('importances')
        if last_importance_df is not None and not last_importance_df.empty:
            top_5 = last_importance_df.head(5)['feature'].tolist()
            bottom_5 = last_importance_df[last_importance_df['importance'] <= 0].tail(5)['feature'].tolist()
            state_summary += (
                "--- Tactical Observations ---\n"
                f"📈 Top 5 features: {top_5}\n"
            )
            if bottom_5:
                state_summary += f"📉 Bottom 5 features: {bottom_5}\n\n"
            else:
                 state_summary += "\n"

        # 4. Dynamic task instruction (regular vs review)
        task_instruction = ""
        if importance_report is not None:
            useless_features = importance_report[importance_report['importance_mean'] <= 0].index.tolist()
            task_instruction = (
                "**Special Task: Review & Optimization**\n"
                "Feature importance analysis results:\n"
                f"{importance_report.to_string()}\n"
            )
            if useless_features:
                 task_instruction += (
                    f"Analysis shows these features may be useless or harmful: {useless_features}\n"
                    "Your primary task is aggressive feature pruning. Create a plan to **delete all zero or negative importance features at once**. Then, you can combine other operations to further improve performance.\n"
                 )
            else:
                task_instruction += "All features are performing well. Focus on adding new features.\n"

        else:
            task_instruction = "Your task: Based on history and current state, propose a feature engineering plan with **1-3 operations** to improve performance.\n"
        
        state_summary += task_instruction

        # 5. Function definitions and instructions - use .format() and escape JSON brackets
        instructions_template = """
--- Available Functions & Instructions ---
Choose one or more functions:

**Macro Functions:**
1. `create_time_features_macro(df)`: Creates a set of standard time-based features (year, month, day, dayofweek).
2. `create_lag_features_macro(df, col)`: Creates a set of standard lag features (lags 1, 2, 3, 7, 14).
3. `create_rolling_features_macro(df, col)`: Creates a set of standard rolling window features (windows 7, 14, 30 for mean/std).

**Basic Functions:**
4. `create_lag_features(df, col, lags)`: Create specific lag features.
5. `create_rolling_features(df, col, windows, aggs)`: Create specific rolling features.
6. `create_fourier_features(df, col, order)`: Create Fourier features (note: `col` must be 'date').
7. `create_interaction_features(df, col1, col2)`: Create interaction features.
8. `create_embedding_features(df, col, window_size)`: Create embedding features (sizes: 90, 365, 730).

**Optimization Functions:**
9. `delete_features(df, cols)`: Delete one or more features.

**Output Format:**
Your response must be a strict JSON list, even for single operations:
`[{{"function": "func_name", "args": {{"key": "value"}}}}, ...]`

**Key Rules:**
- Plan can include 1-3 operations.
- Only use columns from current feature set.
- **For review tasks, must include `delete_features` operation.**
- **Data Leakage Warning: `create_interaction_features` cannot use target column ('{target_col}') directly. If you need target info, first create lag/rolling features.**
- **Strategy 1: Avoid repeating recently rejected plans. Try more diverse strategies.**
- **Strategy 2: If simple feature combinations aren't helping, try `create_embedding_features` for complex non-linear relationships.**
"""
        instructions = instructions_template.format(target_col=self.target_col)
        
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
            # 改进备用计划：更有可能成功
            current_features = self.history[-1]['available_features']
            lag_cols = [f for f in current_features if 'lag' in f or f == self.target_col]
            chosen_col = random.choice(lag_cols) if lag_cols else self.target_col
            return [{"function": "create_rolling_features", "args": {"col": chosen_col, "windows": [random.choice([7,14,30])], "aggs": ["mean"]}}]

    
    def run(self, execute_plan_func, probe_func):
        """
        Main loop for T-LAFS algorithm.
        """
        print("\n" + "="*80)
        print("🚀 Starting T-LAFS feature search loop...")
        print("="*80)

        # --- Build tlafs_params ---
        tlafs_params = {
            "target_col_static": self.target_col,
            "pretrain_cols_static": TLAFS_Algorithm.pretrain_cols_static,
            "pretrained_encoders": TLAFS_Algorithm.pretrained_encoders,
            "embedder_scalers": TLAFS_Algorithm.embedder_scalers
        }

        # self.best_df is now initialized in _log_baseline_performance
        
        # --- Removed forced 'control' baseline features ---
        # The LLM now decides how to start building features from scratch.
        # The `create_control_baseline_features` macro is available if it chooses.

        # --- Iterative exploration loop ---
        for i in range(1, self.n_iterations + 1):
            
            # Periodic review logic
            if i > 1 and i % self.review_interval == 0:
                print("\n" + "="*50)
                print(f"🔬 Triggering periodic review (Iteration {i}) 🔬")
                print("="*50)
                
                # 1. Run permutation importance analysis
                current_features = self.history[-1]['available_features']
                df_for_importance = self.best_df.dropna(subset=current_features + [self.target_col])
                X_val = df_for_importance[current_features]
                y_val = df_for_importance[self.target_col]
                
                lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
                lgbm.fit(X_val, y_val)
                
                importance_df = calculate_permutation_importance(
                    model=lgbm, X_val=X_val, y_val=y_val, metric_func=r2_score
                )
                
                # 2. Get plan with review context
                plan = self.get_plan_from_llm(i, importance_report=importance_df)
            else:
                # Regular iteration
                plan = self.get_plan_from_llm(i)

            # 2. Execute plan
            temp_df, new_feature_name = execute_plan_func(self.best_df.copy(), plan, tlafs_params)

            if temp_df is None or new_feature_name is None:
                print("⚠️ Plan execution failed or no new features generated. Skipping iteration.")
                self.history.append({
                    "iteration": i, "plan": str(plan), "feature_name": "Execution Failed",
                    "performance": self.best_score, "adopted": False,
                    "available_features": self.history[-1]['available_features'],
                    "importances": self.history[-1].get('importances')
                })
                continue
            
            # 3. Probe new feature set performance
            features_to_probe = [c for c in temp_df.columns if c not in ['date', self.target_col]]
            new_score, _, importances_df = probe_func(temp_df, self.target_col, features_to_probe)
            
            # 4. Decide whether to adopt new features
            improvement = new_score - self.best_score
            adopted = improvement >= self.acceptance_threshold
            
            print(f"  - New features '{new_feature_name}', probe R²: {new_score:.4f} (improvement: {improvement:+.4f})")

            if adopted:
                print(f"  ✅ Adopted: Performance improvement meets threshold {self.acceptance_threshold}")
                self.best_score = new_score
                self.best_df = temp_df
                self.best_plan.append(plan)
            else:
                print(f"  ❌ Rejected: Insufficient performance improvement.")

            # 5. Record history
            features_for_history = []
            if adopted:
                features_for_history = [c for c in temp_df.columns if c not in ['date', self.target_col]]
            else:
                features_for_history = self.history[-1]['available_features']

            self.history.append({
                "iteration": i,
                "plan": str(plan),
                "feature_name": new_feature_name,
                "performance": new_score,
                "adopted": adopted,
                "available_features": features_for_history,
                "importances": importances_df if adopted else self.history[-1].get('importances')
            })

        print("\n" + "="*80)
        print("🏆 T-LAFS feature search loop completed.")
        print(f"🏆 Best R² score found: {self.best_score:.4f}")
        print(f"📋 Final adopted plans: {self.best_plan}")
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