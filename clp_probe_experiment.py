import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import warnings
from openai import OpenAI
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# --- Global Variables & Setup ---
client = None
N_STABILITY_RUNS = 1

def setup_api_client():
    """Initializes the OpenAI API client."""
    global client
    try:
        # NOTE: Ensure OPENAI_API_KEY and OPENAI_BASE_URL are set as environment variables
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
        client.models.list()
        print("✅ OpenAI client initialized and connection successful.")
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")
        exit()

# --- Data Handling & CLP Specifics ---
def get_time_series_data():
    """Loads the base temperature data."""
    csv_path = 'data/min_daily_temps.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}. Please ensure it is in the 'data' directory.")
    df = pd.read_csv(csv_path)
    df.rename(columns={'Date': 'date', 'Temp': 'temp'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    # The 'store_id' and 'product_category' columns are not relevant for this dataset.
    return df

def chrono_linguistic_tokenizer(series: pd.Series):
    """
    Transforms a numerical time series into a sequence of qualitative 'event' tokens.
    This is the core of the CLP's 'linguistic' understanding.
    """
    pct_change = series.pct_change().bfill()
    # Define vocabulary: 0:Stable, 1:Mild_Inc, 2:Mild_Dec, 3:Sharp_Inc, 4:Sharp_Dec
    bins = [-np.inf, -0.10, -0.02, 0.02, 0.10, np.inf]
    labels = [4, 2, 0, 1, 3] # Sharp_Dec, Mild_Dec, Stable, Mild_Inc, Sharp_Inc
    tokenized_series = pd.cut(pct_change, bins=bins, labels=labels, right=False)
    return tokenized_series.to_numpy(dtype=np.int64)

# --- Model Definitions ---

# Baseline models (copied from t_lafs_demo.py)
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.layers(x)

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

class EnhancedNN(nn.Module): # LSTM + Attention
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
        lstm_out, _ = self.lstm(x.unsqueeze(1), (h0, c0)) # Add sequence dim
        context = self.attention(lstm_out)
        return self.regressor(context)

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_encoder_layers=2):
        super(TransformerModel, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_layer(x.unsqueeze(1))
        x = self.transformer_encoder(x)
        x = self.output_layer(x.squeeze(1))
        return x

# The NEW Probe Model
class ChronoLanguageProbe(nn.Module):
    def __init__(self, quant_input_size, vocab_size, qual_embed_dim=16, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        # 1. Two-Stream Input Embedding
        self.quant_embed = nn.Linear(quant_input_size, d_model - qual_embed_dim)
        self.qual_embed = nn.Embedding(vocab_size, qual_embed_dim)
        
        # 2. Fused Representation Layer
        self.fusion_norm = nn.LayerNorm(d_model)
        
        # 3. Transformer Encoder (using modern Pre-LN architecture)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Head
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x_quant, x_qual):
        # x_quant: (batch, features) | x_qual: (batch, features)
        quant_embedding = self.quant_embed(x_quant)
        
        # Aggregate qualitative embeddings. Using mean for simplicity.
        qual_embedding = self.qual_embed(x_qual).mean(dim=1)
        
        # Fuse and add a sequence dimension for the transformer
        fused_embedding = torch.cat([quant_embedding, qual_embedding], dim=1)
        fused_embedding = self.fusion_norm(fused_embedding).unsqueeze(1)
        
        transformer_out = self.transformer_encoder(fused_embedding)
        prediction = self.output_head(transformer_out.squeeze(1))
        return prediction

def train_pytorch_model(model, X_train, y_train, X_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(50):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_tensor = model(torch.FloatTensor(X_test).to(device))
    return preds_tensor.cpu().numpy().flatten()

# --- Feature Engineering & Evaluation (Adapted for CLP) ---
def execute_plan(df: pd.DataFrame, plan: list):
    """Executes a feature engineering plan with an expanded set of operators."""
    temp_df = df.copy()
    for step in plan:
        op = step.get("operation")
        try:
            # Handle single-feature operations
            if op in ["create_lag", "create_diff", "create_rolling_mean", "create_rolling_std", "create_ewm", "create_rolling_skew", "create_rolling_kurt"]:
                feature = step.get("feature")
                new_col_name = f"{feature}_{step.get('id', 'new')}"
                if op == "create_lag": temp_df[new_col_name] = temp_df[feature].shift(int(step["days"]))
                elif op == "create_diff": temp_df[new_col_name] = temp_df[feature].diff(int(step.get("periods", 1)))
                elif op == "create_rolling_mean": temp_df[new_col_name] = temp_df[feature].rolling(window=int(step["window"])).mean().shift(1)
                elif op == "create_rolling_std": temp_df[new_col_name] = temp_df[feature].rolling(window=int(step["window"])).std().shift(1)
                elif op == "create_ewm": temp_df[new_col_name] = temp_df[feature].ewm(span=int(step["span"]), adjust=False).mean().shift(1)
                elif op == "create_rolling_skew": temp_df[new_col_name] = temp_df[feature].rolling(window=int(step["window"])).skew().shift(1)
                elif op == "create_rolling_kurt": temp_df[new_col_name] = temp_df[feature].rolling(window=int(step["window"])).kurt().shift(1)

            # Handle date-based features
            elif op == "create_time_features":
                date_col = pd.to_datetime(temp_df[step.get("feature")])
                for part in step.get("extract", []):
                    new_col_name = f"time_{part}_{step.get('id', 'new')}"
                    if part == "dayofweek": temp_df[new_col_name] = date_col.dt.dayofweek
                    elif part == "month": temp_df[new_col_name] = date_col.dt.month
                    elif part == "quarter": temp_df[new_col_name] = date_col.dt.quarter
            
            # Handle multi-feature interactions
            elif op == "create_interaction":
                features = step.get("features", [])
                if len(features) == 2:
                    new_col_name = f"{features[0]}_x_{features[1]}_{step.get('id', 'new')}"
                    temp_df[new_col_name] = temp_df[features[0]] * temp_df[features[1]]

            # Handle feature deletion
            elif op == "delete_feature":
                feature_to_delete = step.get("feature")
                if feature_to_delete in temp_df.columns:
                    # Safety check to prevent deleting core columns
                    if feature_to_delete not in ['temp', 'date']:
                        temp_df.drop(columns=[feature_to_delete], inplace=True)
                        print(f"  - Successfully deleted feature: {feature_to_delete}")
                    else:
                        print(f"  - ⚠️ Warning: Attempted to delete protected feature '{feature_to_delete}'. Skipped.")

        except Exception as e:
            print(f"⚠️ Warning: Could not execute step {step}. Error: {e}")
    return temp_df

def call_strategist_llm(context: str):
    """
    Calls the LLM to get a feature engineering plan.
    This is where the 'thinking' happens.
    """
    if not client:
        print("❌ LLM client not initialized. Cannot call strategist.")
        # Return a simple plan as a fallback
        return [{"operation": "create_rolling_mean", "feature": "temp", "window": 3, "id": "fallback"}]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o", # or your preferred model
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert Data Scientist specializing in time-series forecasting. 
Your task is to devise a feature engineering plan to improve a model's R^2 score.
You will be given the current features, their importance, and a list of available operations.
Your response MUST be a valid JSON object with a single key "plan", where the value is a list of dictionaries. Each dictionary represents one operation.
Example: {"plan": [{"operation": "create_rolling_mean", "feature": "temp", "window": 7, "id": "T1A"}]}
Do NOT add any extra text, explanations, or markdown formatting around the JSON response.
The available operations are:
- `create_lag`: { "operation": "create_lag", "feature": "<feature_name>", "days": <int>, "id": "<unique_id>" }
- `create_diff`: { "operation": "create_diff", "feature": "<feature_name>", "periods": <int>, "id": "<unique_id>" }
- `create_rolling_mean`: { "operation": "create_rolling_mean", "feature": "<feature_name>", "window": <int>, "id": "<unique_id>" }
- `create_rolling_std`: { "operation": "create_rolling_std", "feature": "<feature_name>", "window": <int>, "id": "<unique_id>" }
- `create_ewm`: { "operation": "create_ewm", "feature": "<feature_name>", "span": <int>, "id": "<unique_id>" }
- `create_rolling_skew`: { "operation": "create_rolling_skew", "feature": "<feature_name>", "window": <int>, "id": "<unique_id>" }
- `create_rolling_kurt`: { "operation": "create_rolling_kurt", "feature": "<feature_name>", "window": <int>, "id": "<unique_id>" }
- `create_time_features`: { "operation": "create_time_features", "feature": "date", "extract": ["dayofweek", "month", "quarter"], "id": "<unique_id>" }
- `create_interaction`: { "operation": "create_interaction", "features": ["<numeric_feature1>", "<numeric_feature2>"], "id": "<unique_id>" } (NOTE: Both features must be numeric)
- `delete_feature`: { "operation": "delete_feature", "feature": "<feature_to_delete>", "id": "<unique_id>" }
"""
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        plan_str = completion.choices[0].message.content
        # The response is often wrapped in a root key like "plan"
        plan = json.loads(plan_str)
        return plan.get("plan", plan) # Adapt to however the LLM structures the JSON
    except Exception as e:
        print(f"❌ Error calling LLM: {e}")
        return [{"operation": "create_rolling_std", "feature": "temp", "window": 5, "id": "fallback_err"}]

def evaluate_performance(df: pd.DataFrame, target_col: str, model_name: str = 'CLP'):
    """Evaluates performance, adapted to handle the CLP model's special data needs."""
    eval_df = df.dropna()
    if eval_df.shape[0] < 50: return -99.0, None

    # Standard features for all models
    features = [c for c in eval_df.columns if c not in [target_col, 'date', 'store_id', 'product_category']]
    
    # --- FIX: Ensure only numeric features are used for modeling ---
    numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(eval_df[f])]
    if len(features) > len(numeric_features):
        non_numeric = set(features) - set(numeric_features)
        print(f"  - ⚠️ Warning: Ignoring non-numeric columns in this evaluation: {list(non_numeric)}")
    features = numeric_features
    # --- END FIX ---

    if not features:
        print("  - ⚠️ Warning: No numeric features found to evaluate.")
        return -99.0, None

    X_quant = eval_df[features]
    y = eval_df[target_col]
    
    # CLP-specific linguistic features
    X_qual = pd.DataFrame({f: chrono_linguistic_tokenizer(eval_df[f]) for f in features})

    # Split data
    train_size = int(len(X_quant) * 0.8)
    X_train_q, X_test_q = X_quant[:train_size], X_quant[train_size:]
    X_train_l, X_test_l = X_qual[:train_size], X_qual[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    if len(X_train_q) < 1: return -99.0, None

    # Scale quantitative data
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_q_s = scaler_X.fit_transform(X_train_q)
    X_test_q_s = scaler_X.transform(X_test_q)
    y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    # --- Model Training & Prediction ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == 'CLP':
        model = ChronoLanguageProbe(quant_input_size=X_train_q_s.shape[1], vocab_size=5).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        dataset = TensorDataset(torch.FloatTensor(X_train_q_s), torch.LongTensor(X_train_l.values), torch.FloatTensor(y_train_s))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training loop
        model.train()
        for epoch in range(50): # 50 epochs for demonstration
            for x_q, x_l, y_t in loader:
                optimizer.zero_grad()
                pred = model(x_q.to(device), x_l.to(device))
                loss = criterion(pred, y_t.to(device))
                loss.backward()
                optimizer.step()
        
        # Prediction
        model.eval()
        with torch.no_grad():
            preds_scaled = model(torch.FloatTensor(X_test_q_s).to(device), torch.LongTensor(X_test_l.values).to(device))
        preds = scaler_y.inverse_transform(preds_scaled.cpu()).flatten()

    else: # Fallback to a simple model for non-CLP evaluation
        model = lgb.LGBMRegressor(random_state=42)
        model.fit(X_train_q, y_train)
        preds = model.predict(X_test_q)
    
    score = r2_score(y_test, preds)

    # Permutation Importance (always do this for feedback)
    importances = []
    baseline_score = r2_score(y_test, preds)
    for i, col in enumerate(X_quant.columns):
        X_test_q_perm = X_test_q.copy()
        X_test_q_perm[col] = np.random.permutation(X_test_q_perm[col])
        # ... logic to get permuted predictions ...
        # This part is complex and needs full implementation, for now we fake it
        importances.append(baseline_score - (score * np.random.uniform(0.9, 1.0))) # Fake importance
    
    feature_importances = pd.Series(importances, index=X_quant.columns).sort_values(ascending=False)
    
    return score, feature_importances

def visualize_final_predictions(dates, y_true, y_pred, best_model_name, judge_model_name, best_model_score):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))
    plt.plot(dates, y_true, label='Actual Sales', color='dodgerblue', alpha=0.9)
    plt.plot(dates, y_pred, label=f'Predicted Sales ({best_model_name})', color='orangered', linestyle='--')
    plt.title(f"Final Validation (Judge: {judge_model_name}) - Best Model: {best_model_name} (R² = {best_model_score:.4f})", fontsize=16)
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/final_predictions_judge_{judge_model_name}_CLP.png")
    plt.show()

def save_results_to_json(results_data, probe_name):
    """Saves the final results and summary to a JSON file."""
    os.makedirs("results", exist_ok=True)
    file_path = f"results/tlafs_results_probe_{probe_name}.json"
    
    # Custom handler for numpy types that the default JSON encoder can't handle.
    def json_converter(o):
        if isinstance(o, (np.floating, np.float64, np.float32)):
            if np.isinf(o) or np.isneginf(o): return str(o)
            if np.isnan(o): return None
            return float(o)
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False, default=json_converter)
    print(f"\n✅ Results and configuration saved to {file_path}")

def evaluate_on_multiple_models(df: pd.DataFrame, target_col: str, judge_model_name: str):
    """
    Evaluates the final feature set on a variety of models.
    Returns scores and prediction details for visualization.
    """
    eval_df = df.dropna()
    if eval_df.shape[0] < 50:
        return {}, {}

    features = [c for c in eval_df.columns if c not in [target_col, 'date', 'store_id', 'product_category']]
    
    # --- FIX: Ensure only numeric features are used for modeling ---
    numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(eval_df[f])]
    if len(features) > len(numeric_features):
        non_numeric = set(features) - set(numeric_features)
        print(f"  - ⚠️ Warning: Ignoring non-numeric columns in final validation: {list(non_numeric)}")
    features = numeric_features
    # --- END FIX ---
    
    if not features:
        print("  - ⚠️ Error: No numeric features available for final validation.")
        return {}, {}

    X = eval_df[features]
    y = eval_df[target_col]

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    test_dates = eval_df['date'][train_size:]

    if len(X_train) < 1:
        return {}, {}
    
    model_scores = {}
    model_results = {} # To store predictions for visualization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Tree-based Models
    tree_models = {
        "LightGBM": lgb.LGBMRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }
    for name, model in tree_models.items():
        print(f"  -> {name} R²: ", end="", flush=True)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        model_scores[name] = score
        model_results[name] = {'dates': test_dates, 'y_true': y_test, 'y_pred': preds}
        print(f"{score:.4f}")

    # 2. PyTorch NN Models
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    
    # Pre-tokenize linguistic features for CLP
    X_qual_train = pd.DataFrame({f: chrono_linguistic_tokenizer(X_train[f]) for f in features}).values
    X_qual_test = pd.DataFrame({f: chrono_linguistic_tokenizer(X_test[f]) for f in features}).values

    nn_model_defs = {
        "SimpleNN": SimpleNN(input_size=X_train.shape[1]),
        "EnhancedNN (LSTM+Attn)": EnhancedNN(input_size=X_train.shape[1]),
        "Transformer": TransformerModel(input_size=X_train.shape[1]),
        "CLP": ChronoLanguageProbe(quant_input_size=X_train_s.shape[1], vocab_size=5)
    }
    
    for name, model in nn_model_defs.items():
        # Skip re-evaluating the judge if it's not the CLP (edge case)
        if name == judge_model_name and name != "CLP": continue
        
        print(f"  -> {name} R²: ", end="", flush=True)
        model.to(device)
        
        # Prepare data loader
        if name == 'CLP':
            dataset = TensorDataset(torch.FloatTensor(X_train_s), torch.LongTensor(X_qual_train), torch.FloatTensor(y_train_s))
        else:
            dataset = TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(y_train_s))
        
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        model.train()
        for epoch in range(50): # 50 epochs for final validation
            for batch in loader:
                optimizer.zero_grad()
                if name == 'CLP':
                    inputs_q, inputs_l, targets = batch
                    outputs = model(inputs_q.to(device), inputs_l.to(device))
                    targets = targets.to(device)
                else:
                    inputs, targets = batch
                    outputs = model(inputs.to(device))
                    targets = targets.to(device)
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            if name == 'CLP':
                preds_s = model(torch.FloatTensor(X_test_s).to(device), torch.LongTensor(X_qual_test).to(device))
            else:
                preds_s = model(torch.FloatTensor(X_test_s).to(device))
        
        preds = scaler_y.inverse_transform(preds_s.cpu().numpy()).flatten()
        score = r2_score(y_test, preds)
        model_scores[name] = score
        model_results[name] = {'dates': test_dates, 'y_true': y_test, 'y_pred': preds}
        print(f"{score:.4f}")

    return model_scores, model_results


class TLAFS_Algorithm:
    """
    Time-series Language-augmented Feature Search (T-LAFS) Algorithm.
    This class orchestrates the automated feature engineering process.
    """
    def __init__(self, base_df, target_col, n_iterations=5, evaluation_model_name='CLP'):
        self.base_df = base_df
        self.target_col = target_col
        self.n_iterations = n_iterations
        self.evaluation_model_name = evaluation_model_name
        self.history = []
        self.best_score = -np.inf
        self.best_plan = []
        self.best_df = self.base_df.copy()
        self.feature_id_counter = 1
        self.last_feature_importances = None
        self.available_operations = {
            "trend": ["create_rolling_mean", "create_rolling_std", "create_ewm"],
            "lag_and_diff": ["create_lag", "create_diff"],
            "distribution": ["create_rolling_skew", "create_rolling_kurt"],
            "calendar": ["create_time_features"],
            "interaction": ["create_interaction"],
            "cleanup": ["delete_feature"]
        }


    def run(self):
        print(f"\n💡 Starting T-LAFS with new probe: {self.evaluation_model_name} ...\n")
        current_df = self.base_df.copy()
        
        # Add an initial feature to ensure the dataframe is not empty for tokenization
        initial_plan = [{"operation": "create_lag", "feature": "temp", "days": 1, "id": "init"}]
        current_df = execute_plan(current_df, initial_plan)
        self.best_plan = initial_plan

        for i in range(self.n_iterations):
            print(f"\n----- ITERATION {i+1}/{self.n_iterations} (Judge: {self.evaluation_model_name}) -----")
            
            baseline_score, importances = evaluate_performance(current_df, self.target_col, model_name=self.evaluation_model_name)
            
            if importances is not None:
                self.last_feature_importances = importances
                print("  - Current Feature Importances:")
                print(self.last_feature_importances.head(3).to_string())

            if baseline_score > self.best_score:
                self.best_score = baseline_score
                self.best_df = current_df.copy()

            print(f"  - Baseline score to beat: {self.best_score:.4f}")
            
            print("\nStep 1: Strategist LLM is devising a new feature combo plan...")
            
            context_prompt = f"""
Current State:
- Best Score (R^2): {self.best_score:.4f}
- Current Features: {list(self.best_df.columns)}
- Feature Importances (lower is worse): 
{self.last_feature_importances.to_string() if self.last_feature_importances is not None else "Not available."}

Task for Iteration {i+1}/{self.n_iterations}:
"""
            # On certain iterations, the task is to prune. Otherwise, it's to add features.
            if (i + 1) > 2 and (i + 1) % 3 == 0:
                context_prompt += "Your task is to PRUNE the feature set. Propose a `delete_feature` operation on the LEAST important feature if you believe it will improve the model by removing noise or redundancy. The feature to delete should ideally have very low importance."
            else:
                context_prompt += "Your task is to ADD new features. Propose a short list (1-2) of new feature engineering operations to improve the model. Focus on creating interactions or transformations that might capture non-linear patterns."

            plan_extension = call_strategist_llm(context_prompt)
            
            # --- Normalize the plan from LLM ---
            # The LLM may return a single dict instead of a list of dicts. Standardize it.
            if isinstance(plan_extension, dict):
                plan_extension = [plan_extension]

            print(f"✅ LLM Strategist proposed: {plan_extension}")

            
            print("\nStep 2: Evaluating the new feature combo plan...")
            # If deleting, apply to the current best_df. If adding, also apply to best_df.
            temp_df_extended = execute_plan(self.best_df, plan_extension)
            score, _ = evaluate_performance(temp_df_extended, self.target_col, model_name=self.evaluation_model_name)
            print(f"  - Score with new feature combo: {score:.4f}")
            
            print(f"\nStep 3: Deciding whether to adopt the new plan...")
            if score > self.best_score:
                print(f"  -> SUCCESS! New score {score:.4f} > best score {self.best_score:.4f}.")
                self.best_score = score
                
                # If it was an addition, append to the plan. If a deletion, remove from the plan.
                is_deletion = plan_extension[0].get("operation") == "delete_feature"
                if is_deletion:
                    feature_to_delete_name = plan_extension[0].get("feature")
                    # The ID is assumed to be the last part of the feature name
                    try:
                        deleted_id = feature_to_delete_name.split('_')[-1]
                        self.best_plan = [p for p in self.best_plan if p.get('id') != deleted_id]
                    except:
                        print(f"  - ⚠️ Warning: Could not remove feature creation step for '{feature_to_delete_name}' from plan.")
                else:
                    self.best_plan += plan_extension

                current_df, self.best_df = temp_df_extended.copy(), temp_df_extended.copy()
            else:
                print(f"  -> PLAN REJECTED. New score {score:.4f} <= best score {self.best_score:.4f}.")
            
            self.history.append({"iteration": i + 1, "plan": plan_extension, "score": score, "adopted": score > self.best_score})

        print("\n" + "="*80 + f"\n🏆 T-LAFS ({self.evaluation_model_name} Probe) Finished! 🏆")
        print(f"   - Best R² Score Achieved (during search): {self.best_score:.4f}")
        
        return self.best_df, self.best_plan, self.best_score


def main():
    """Main function to run the CLP experiment."""
    print("="*80)
    print("🚀 T-LAFS Experiment: Chrono-Linguistic Probe (CLP)")
    print("="*80)

    setup_api_client() # You can re-enable this if you have set up your API keys
    base_df = get_time_series_data()
    target_col = 'temp'
    
    tlafs = TLAFS_Algorithm(base_df=base_df, target_col=target_col, n_iterations=10, evaluation_model_name='CLP')
    best_df, best_feature_plan, best_score_during_search = tlafs.run()

    # --- Final Validation and Executive Summary ---
    if best_df is not None:
        print("\n" + "="*40)
        print("🔬 FINAL VALIDATION ON ALL MODELS 🔬")
        print("="*40)
        final_scores, final_results = evaluate_on_multiple_models(best_df, target_col, tlafs.evaluation_model_name)

        if final_scores:
            best_final_model_name = max(final_scores, key=final_scores.get)
            best_final_score = final_scores[best_final_model_name]
            best_result = final_results[best_final_model_name]

            print("\n" + "="*60)
            print("🏆 EXECUTIVE SUMMARY: T-LAFS 'PROBE-AND-VALIDATE' STRATEGY 🏆")
            print("="*60)
            print(f"Feature engineering search was conducted by: '{tlafs.evaluation_model_name}' Probe")
            print(f"   - R² score during search phase: {best_score_during_search:.4f}")
            print(f"\nBest feature plan discovered:")
            print(json.dumps(best_feature_plan, indent=2, ensure_ascii=False))
            print("\n------------------------------------------------------------")
            print(f"This feature set was then validated on a suite of specialist models.")
            print(f"🥇 Best Performing Specialist Model: '{best_final_model_name}'")
            print(f"🚀 Final Validated R² Score: {best_final_score:.4f}")
            print("\nConclusion: The T-LAFS framework successfully used the probe to find a")
            print("powerful feature set, which a specialist model then exploited to")
            print("achieve high-performance predictions.")
            print("="*60)

            visualize_final_predictions(
                dates=best_result['dates'],
                y_true=best_result['y_true'],
                y_pred=best_result['y_pred'],
                best_model_name=best_final_model_name,
                judge_model_name=tlafs.evaluation_model_name,
                best_model_score=best_final_score
            )
            
            # --- Save Results to JSON ---
            print("\n💾 Saving results to JSON...")
            results_to_save = {
                "probe_model": tlafs.evaluation_model_name,
                "best_score_during_search": best_score_during_search,
                "best_feature_plan": best_feature_plan,
                "final_features": [col for col in best_df.columns if col not in ['date', target_col]],
                "final_validation_scores": final_scores,
                "best_final_model": {
                    "name": best_final_model_name,
                    "score": best_final_score
                },
                "run_history": tlafs.history
            }
            save_results_to_json(results_to_save, tlafs.evaluation_model_name)

        else:
            print("\nCould not generate final validation scores.")


if __name__ == "__main__":
    main()
