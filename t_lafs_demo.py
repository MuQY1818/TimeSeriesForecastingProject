import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import warnings
from openai import OpenAI
import os
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
# import shap

# --- PyTorch Imports for Neural Networks ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

# --- Global Variables & Setup ---
client = None
N_STABILITY_RUNS = 1  # Will be set in main

def setup_api_client():
    """Initializes the OpenAI API client by reading from environment variables."""
    global client
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        print("✅ OpenAI client initialized successfully.")
        print("   - Checking connection to API...")
        client.models.list()
        print("   - Connection successful.")
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")
        print("Please ensure your OPENAI_API_KEY and (optional) OPENAI_BASE_URL environment variables are set correctly.")
        exit()

def print_welcome():
    """Prints a welcome message for the demo."""
    print("="*80)
    print("🚀 Welcome to the T-LAFS (Time-series LLM-driven Adaptive Feature Synthesis) Demo!")
    print("This script will now use real data from 'total_cleaned.csv' and demonstrate how")
    print("an AI can iteratively propose and evaluate new features to improve a forecast model.")
    print("="*80)

# --- Data Handling ---

def get_time_series_data():
    """
    Loads real sales data, adding mock features for a more realistic scenario.
    """
    csv_path = 'data/total_cleaned.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}.")
        
    df = pd.read_csv(csv_path, encoding='gbk', header=None, skiprows=1, names=['date', 'sales'])
    
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    
    df['store_id'] = 'Store_A'
    df['product_category'] = 'Category_X'
    
    return df

# --- Neural Network Model Definitions ---

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.layers(x)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        attn_weights = self.attn(lstm_output).squeeze(2)
        soft_attn_weights = self.softmax(attn_weights)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context

class EnhancedNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(EnhancedNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attention = Attention(hidden_size)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        context = self.attention(lstm_out)
        output = self.regressor(context)
        return output

def train_pytorch_model(model, X_train, y_train, X_test, y_test, is_lstm=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    if is_lstm:
        X_train_tensor = X_train_tensor.unsqueeze(1)
        X_test_tensor = X_test_tensor.unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(50):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)
        predictions = predictions_tensor.cpu().numpy().flatten()
        
    return predictions

# --- LLM and Feature Engineering ---

def real_llm_call(prompt: str, system_message: str):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
            model="gpt-4o",
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"❌ LLM API call failed: {e}")
        return json.dumps({"error": str(e), "plan": [], "rationale": "An error occurred."})

def execute_plan(df: pd.DataFrame, plan: list):
    temp_df = df.copy()
    for step in plan:
        op = step.get("operation")
        feature = step.get("feature")
        new_col_name = f"{feature}_{step.get('id', 'new')}"
        
        try:
            if op == "create_lag":
                temp_df[new_col_name] = temp_df[feature].shift(int(step["days"]))
            elif op == "create_diff":
                temp_df[new_col_name] = temp_df[feature].diff(int(step.get("periods", 1)))
            elif op == "create_rolling_mean":
                temp_df[new_col_name] = temp_df[feature].rolling(window=int(step["window"])).mean().shift(1)
            elif op == "create_rolling_std":
                temp_df[new_col_name] = temp_df[feature].rolling(window=int(step["window"])).std().shift(1)
            elif op == "create_ewm":
                temp_df[new_col_name] = temp_df[feature].ewm(span=int(step["span"]), adjust=False).mean().shift(1)
            elif op == "create_time_features":
                date_col = pd.to_datetime(temp_df[feature])
                for extract_part in step.get("extract", []):
                    if extract_part == "dayofweek": temp_df[f"time_{extract_part}"] = date_col.dt.dayofweek
                    elif extract_part == "month": temp_df[f"time_{extract_part}"] = date_col.dt.month
                    elif extract_part == "quarter": temp_df[f"time_{extract_part}"] = date_col.dt.quarter
                    elif extract_part == "dayofyear": temp_df[f"time_{extract_part}"] = date_col.dt.dayofyear
        except Exception as e:
            print(f"⚠️ Warning: Could not execute step {step}. Error: {e}")
    return temp_df

# --- Evaluation and Visualization ---

def visualize_final_predictions(dates, y_true, y_pred, best_model_name, judge_model_name, best_model_score):
    """Visualizes the best model's predictions against actual values."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 8))
    
    plt.plot(dates, y_true, label='Actual Sales', color='dodgerblue', alpha=0.9, linewidth=2)
    plt.plot(dates, y_pred, label=f'Predicted Sales ({best_model_name})', color='orangered', linestyle='--', alpha=0.9, linewidth=2.5)
    
    title_text = f"Final Validation (Judge: {judge_model_name}) - Best Model: {best_model_name}\n(R² = {best_model_score:.4f})"
    plt.title(title_text, fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    os.makedirs("plots", exist_ok=True)
    plot_filename = f"plots/final_predictions_judge_{judge_model_name}.png"
    plt.savefig(plot_filename)
    print(f"📊 Final prediction visualization saved to '{plot_filename}'")
    plt.close()

def evaluate_performance(df: pd.DataFrame, target_col: str, model_name: str = 'LightGBM'):
    """Evaluates performance using a specified model, with stability runs for NNs."""
    eval_df = df.copy()

    # --- FIX: One-Hot Encode categorical features, same as in final validation ---
    categorical_cols = eval_df.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        eval_df = pd.get_dummies(eval_df, columns=categorical_cols, dummy_na=False)
    # --- END FIX ---
    
    eval_df = eval_df.dropna()
    if eval_df.shape[0] < 50: return -99.0, None

    X = eval_df.drop(columns=[c for c in [target_col, 'date'] if c in eval_df.columns])
    y = eval_df[target_col]
    
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    if len(X_train) < 1 or len(X_test) < 1: return -99.0, None
    
    feature_importances = None
    preds = None
        
    if model_name == 'LightGBM':
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.01, num_leaves=31, random_state=42, verbosity=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if hasattr(model, 'feature_importances_'):
            feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    elif model_name == 'XGBoost':
        model = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=5, objective='reg:squarederror')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if hasattr(model, 'feature_importances_'):
            feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    elif model_name == 'RandomForest':
        model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10, n_jobs=1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if hasattr(model, 'feature_importances_'):
            feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    elif model_name in ['SimpleNN', 'EnhancedNN (LSTM+Attn)']:
        scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        
        all_preds = []
        # For SHAP, we only need one representative model and its predictions
        final_model_for_shap = None
        for i in range(N_STABILITY_RUNS):
            if model_name == 'SimpleNN': model = SimpleNN(input_size=X_train_scaled.shape[1])
            else: model = EnhancedNN(input_size=X_train_scaled.shape[1])
            preds_scaled = train_pytorch_model(model, X_train_scaled, y_train_scaled.flatten(), X_test_scaled, y_test.values, "LSTM" in model_name)
            all_preds.append(preds_scaled)
            if i == N_STABILITY_RUNS - 1:
                final_model_for_shap = model
        
        avg_preds_scaled = np.mean(all_preds, axis=0)
        preds = scaler_y.inverse_transform(avg_preds_scaled.reshape(-1, 1)).flatten()

        # --- Permutation Importance Calculation for NN Models ---
        if final_model_for_shap:
            final_model_for_shap.eval() # Ensure model is in evaluation mode
            baseline_score = r2_score(y_test, preds)
            importances = []
            device = next(final_model_for_shap.parameters()).device

            for i in range(X_test_scaled.shape[1]):
                X_test_permuted = X_test_scaled.copy()
                np.random.shuffle(X_test_permuted[:, i])
                
                permuted_tensor = torch.FloatTensor(X_test_permuted).to(device)
                if "LSTM" in model_name:
                    permuted_tensor = permuted_tensor.unsqueeze(1)
                    
                with torch.no_grad():
                    permuted_preds_scaled = final_model_for_shap(permuted_tensor).cpu().numpy()

                permuted_preds = scaler_y.inverse_transform(permuted_preds_scaled.reshape(-1, 1)).flatten()
                permuted_score = r2_score(y_test, permuted_preds)
                
                # Importance is the drop in score
                importances.append(baseline_score - permuted_score)
            
            feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    else:
        raise ValueError(f"Unknown model_name '{model_name}' for evaluation.")

    if preds is None:
        return -99.0, None

    score = r2_score(y_test, preds)
    return score, feature_importances

def evaluate_on_multiple_models(df, target_col, judge_model_name):
    """Evaluates the final feature set on all models, saves weights, and visualizes the best."""
    df_eval = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df_eval['date']):
        df_eval['date'] = pd.to_datetime(df_eval['date'])
        
    categorical_cols = df_eval.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        df_eval = pd.get_dummies(df_eval, columns=categorical_cols, dummy_na=False)
        
    df_eval = df_eval.sort_values('date').dropna()
    
    if df_eval.empty: return {} 

    split_date = df_eval['date'].quantile(0.8, interpolation='nearest')
    train_df = df_eval[df_eval['date'] < split_date]
    test_df = df_eval[df_eval['date'] >= split_date]

    if train_df.empty or test_df.empty: return {}
        
    features = [col for col in df_eval.columns if col not in [target_col, 'date']]
    X_train, y_train = train_df[features], train_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]
    test_dates = test_df['date']

    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    
    models = {
        "LightGBM": lgb.LGBMRegressor(random_state=42, n_estimators=100, learning_rate=0.1, num_leaves=31),
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10, n_jobs=1),
        "XGBoost": XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=5, objective='reg:squarederror')
    }
    nn_models = {
        "SimpleNN": SimpleNN(input_size=X_train_scaled.shape[1]),
        "EnhancedNN (LSTM+Attn)": EnhancedNN(input_size=X_train_scaled.shape[1])
    }

    results = {}
    best_model_score, best_model_name, best_model_predictions = -np.inf, "", None

    print("\n" + "="*40 + "\n🔬 FINAL VALIDATION 🔬\n" + "="*40)
    os.makedirs("saved_models", exist_ok=True)
    
    for name, model in models.items():
        print(f"  - Training {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        results[name] = {'r2': r2, 'mae': mae, 'rmse': rmse}
        print(f"    -> {name} R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            print(f"      Top 5 Features for {name}:")
            print(importances.head(5).to_string())
        
        joblib.dump(model, f"saved_models/judge_{judge_model_name}__{name}_model.joblib")
        if r2 > best_model_score: best_model_score, best_model_name, best_model_predictions = r2, name, predictions

    for name, model in nn_models.items():
        print(f"  - Training {name} ({N_STABILITY_RUNS} runs)...")
        all_final_preds = []
        for i in range(N_STABILITY_RUNS):
            print(f"    - Run {i+1}/{N_STABILITY_RUNS}...")
            if name == 'SimpleNN': fresh_model = SimpleNN(input_size=X_train_scaled.shape[1])
            else: fresh_model = EnhancedNN(input_size=X_train_scaled.shape[1])
            predictions_scaled = train_pytorch_model(fresh_model, X_train_scaled, y_train_scaled.flatten(), X_test_scaled, y_test.values, "LSTM" in name)
            all_final_preds.append(predictions_scaled)

        avg_predictions_scaled = np.mean(all_final_preds, axis=0)
        predictions = scaler_y.inverse_transform(avg_predictions_scaled.reshape(-1, 1)).flatten()
        
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        results[name] = {'r2': r2, 'mae': mae, 'rmse': rmse}
        print(f"    -> {name} Avg R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        # Note: We save the last trained NN model. For a fully deterministic result, one would train one final time.
        # This is sufficient for demonstrating the average performance.
        if r2 > best_model_score: best_model_score, best_model_name, best_model_predictions = r2, name, predictions

    print(f"\n🏆 Best performing model in final validation (by R²): {best_model_name} (R² = {best_model_score:.4f})")
    visualize_final_predictions(test_dates, y_test, best_model_predictions, best_model_name, judge_model_name, best_model_score)
    return results

# --- T-LAFS Algorithm Class ---

class TLAFS_Algorithm:
    def __init__(self, base_df, target_col, n_iterations=5, evaluation_model_name='LightGBM'):
        self.base_df = base_df
        self.target_col = target_col
        self.n_iterations = n_iterations
        self.evaluation_model_name = evaluation_model_name
        self.history = []
        self.best_score = -np.inf
        self.best_plan = []
        self.best_df = self.base_df.copy()
        self.feature_id_counter = 1
        self.available_operations = {"lag_and_diff": ["create_lag", "create_diff"], "trend": ["create_rolling_mean", "create_rolling_std", "create_ewm"], "seasonality": ["create_time_features"]}
        self.last_feature_importances = None

    def run(self):
        print("\n" + "="*80 + "\n💡 Starting T-LAFS Algorithm v4 (Feature-Importance-guided Feedback)...\n")
        current_df = self.base_df.copy()
        
        for i in range(self.n_iterations):
            print(f"\n----- ITERATION {i+1}/{self.n_iterations} (Judge: {self.evaluation_model_name}) -----")
            
            baseline_score, importances = evaluate_performance(current_df, self.target_col, model_name=self.evaluation_model_name)
            if importances is not None:
                self.last_feature_importances = importances
                print("  - Current Feature Importances:")
                print(self.last_feature_importances.head(5).to_string())

            if baseline_score > self.best_score:
                self.best_score = baseline_score
                self.best_plan = []
                self.best_df = current_df.copy()
            print(f"  - Baseline score to beat: {self.best_score:.4f}")
            
            print("\nStep 1: Strategist LLM is devising a new feature combo plan...")
            
            importance_prompt_injection = ""
            if self.last_feature_importances is not None:
                top_features = self.last_feature_importances.head(3).index.tolist()
                bottom_features = self.last_feature_importances.tail(3).index.tolist()
                importance_prompt_injection = f"""
                Based on the last run, the most impactful features were {top_features}, while the least impactful were {bottom_features}.
                Your new plan should consider this feedback. Try to build upon what works, or propose alternatives for what doesn't.
                """

            prompt_strategist = f"""
            As a master Time Series Analyst, your task is to propose a feature engineering plan to improve a sales forecasting model.
            The target variable is '{self.target_col}'. Current feature set: {list(current_df.columns)}.
            The history of past trials (higher R² score is better) is: {json.dumps(self.history, indent=2)}.
            {importance_prompt_injection}
            Your available tools are grouped: Lag/Diff: {self.available_operations['lag_and_diff']}, Trend: {self.available_operations['trend']}, Seasonality: {self.available_operations['seasonality']}.
            Your mission: devise a "feature combo" of 2-3 DIVERSE, NEW features.
            Your output MUST be a valid JSON object with a key "feature_combo_plan" (a list of feature creation steps) and "rationale".
            Example: {{"feature_combo_plan": [{{"operation": "create_lag", "feature": "sales", "days": 14, "id": "L{i+1}A"}}], "rationale": "This captures..."}}
            """
            llm_response_str = real_llm_call(prompt_strategist, "You are an expert data scientist specializing in time series feature engineering.")
            try: llm_response = json.loads(llm_response_str)
            except json.JSONDecodeError:
                print(f"❌ LLM response was not valid JSON: {llm_response_str}"); continue

            plan_extension = llm_response.get("feature_combo_plan")
            rationale = llm_response.get("rationale", "No rationale provided.")
            
            if not plan_extension or not isinstance(plan_extension, list):
                print("❌ Could not get a valid feature combo plan from LLM. Skipping iteration."); continue

            print(f"✅ LLM Strategist proposed a new plan with rationale: '{rationale}'")
            print("\nStep 2: Evaluating the new feature combo plan...")
            temp_df_extended = execute_plan(current_df, plan_extension)
            score, _ = evaluate_performance(temp_df_extended, self.target_col, model_name=self.evaluation_model_name)
            
            print(f"  - Score with new feature combo: {score:.4f}")
            print(f"\nStep 3: Deciding whether to adopt the new plan...")
            
            adopted = score > self.best_score
            if adopted:
                print(f"  -> SUCCESS! New combo score {score:.4f} is better than best score {self.best_score:.4f}.")
                for step in plan_extension: step['id'] = f"F{self.feature_id_counter}"; self.feature_id_counter += 1
                self.best_score, self.best_plan, current_df, self.best_df = score, self.best_plan + plan_extension, temp_df_extended.copy(), temp_df_extended.copy()
                print(f"  ✨ New best feature set adopted! Columns are now:\n     {json.dumps(list(current_df.columns))}")
            else:
                print(f"  -> PLAN REJECTED. New combo score {score:.4f} did not beat best score {self.best_score:.4f}.")
            
            self.history.append({"iteration": i + 1, "proposed_plan": plan_extension, "score": score, "adopted": bool(adopted)})

        print("\n" + "="*80 + f"\n🏆 T-LAFS Algorithm Finished! 🏆")
        print(f"   - Best R² Score Achieved (during search with {self.evaluation_model_name}): {self.best_score:.4f}")
        
        final_scores = evaluate_on_multiple_models(self.best_df, self.target_col, self.evaluation_model_name)

        return {
            "judge_model": self.evaluation_model_name,
            "best_score_achieved": self.best_score,
            "final_cross_val_scores": final_scores,
            "best_feature_plan": self.best_plan,
            "final_feature_columns": list(self.best_df.columns)
        }

# --- Main Execution ---

def main():
    """Main function to run the demo."""
    # --- CONFIGURATION ---
    # Options: 'LightGBM', 'RandomForest', 'XGBoost', 'SimpleNN', 'EnhancedNN (LSTM+Attn)'
    SEARCH_MODEL_JUDGE = 'EnhancedNN (LSTM+Attn)' 
    global N_STABILITY_RUNS; N_STABILITY_RUNS = 3
    # --- END CONFIGURATION ---

    setup_api_client()
    print_welcome()

    print("\nStep A: Loading and visualizing time series data...")
    base_df = get_time_series_data()
    print(f"✅ Data loaded. Shape: {base_df.shape}.")
    print("✅ Starting T-LAFS with raw data. No pre-built features.")

    t_lafs_runner = TLAFS_Algorithm(base_df=base_df, target_col='sales', n_iterations=5, evaluation_model_name=SEARCH_MODEL_JUDGE)
    results = t_lafs_runner.run()

    if results:
        results_filename = f"results_{SEARCH_MODEL_JUDGE}.json"
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Experiment results saved to '{results_filename}'")

if __name__ == "__main__":
    main()
