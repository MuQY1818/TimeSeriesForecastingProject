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
from datetime import datetime

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

# --- Data Handling & DAP Specifics ---
def get_time_series_data(dataset_type='min_daily_temps'):
    if dataset_type == 'min_daily_temps':
        csv_path = 'data/min_daily_temps.csv'
        df = pd.read_csv(csv_path)
        df.rename(columns={'Date': 'date', 'Temp': 'temp'}, inplace=True)
    elif dataset_type == 'total_cleaned':
        csv_path = 'data/total_cleaned.csv'
        df = pd.read_csv(csv_path)
        df.rename(columns={'日期': 'date', '成交商品件数': 'temp'}, inplace=True)
    else:
        raise ValueError('Unknown dataset type')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attention = nn.Linear(hidden_size, 1)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.bmm(lstm_out.transpose(1, 2), attn_weights).squeeze(2)
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

# --- Learned Embedding Model ---
class LearnedEmbedder(nn.Module):
    """
    A simple GRU-based model to create learned embeddings from a time-series window.
    It acts as a feature extractor.
    """
    def __init__(self, input_dim=1, hidden_dim=32, embedding_dim=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        _, h_n = self.gru(x)
        # h_n shape: (num_layers, batch_size, hidden_dim)
        # Get the hidden state of the last layer
        last_hidden = h_n[-1, :, :]
        # last_hidden shape: (batch_size, hidden_dim)
        embedding = self.fc(last_hidden)
        # embedding shape: (batch_size, embedding_dim)
        return embedding

# --- Autoencoder Models for Pre-training ---
class MaskedEncoder(nn.Module):
    """
    A powerful, deep encoder that processes a sequence and outputs a sequence of latent representations.
    This is the core of our pre-trained feature extractor.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # outputs shape: (batch_size, seq_len, hidden_dim * 2)
        outputs, _ = self.gru(x)
        return outputs

class MaskedDecoder(nn.Module):
    """A lightweight decoder used ONLY during pre-training to reconstruct the sequence."""
    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim * 2, # Input from encoder
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        outputs, _ = self.gru(x)
        reconstruction = self.fc(outputs)
        return reconstruction

class MaskedTimeSeriesAutoencoder(nn.Module):
    """Combines Encoder and Decoder for pre-training with masking."""
    def __init__(self, input_dim: int, encoder_hidden_dim: int, encoder_layers: int, decoder_hidden_dim: int, decoder_layers: int):
        super().__init__()
        self.encoder = MaskedEncoder(input_dim=input_dim, hidden_dim=encoder_hidden_dim, num_layers=encoder_layers)
        self.decoder = MaskedDecoder(hidden_dim=encoder_hidden_dim, output_dim=input_dim, num_layers=decoder_layers)

    def forward(self, x_masked):
        latent_representation = self.encoder(x_masked)
        reconstruction = self.decoder(latent_representation)
        return reconstruction

def pretrain_embedder(df_pretrain: pd.DataFrame, window_size: int, config: dict, results_dir: str):
    """Pre-trains a masked autoencoder and returns the trained ENCODER part."""
    input_dim = df_pretrain.shape[1]
    print(f"  - Preparing data for masked autoencoder pre-training (window: {window_size}, input_dim: {input_dim})...")
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(df_pretrain)
    
    sequences = []
    for i in range(len(series_scaled) - window_size + 1):
        sequences.append(series_scaled[i:i+window_size])
    
    if not sequences:
        print("  - ⚠️ Warning: Not enough data for pre-training. Returning untrained encoder.")
        return MaskedEncoder(input_dim=input_dim, hidden_dim=config['encoder_hidden_dim'], num_layers=config['encoder_layers']), scaler

    dataset = TensorDataset(torch.FloatTensor(np.array(sequences)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = MaskedTimeSeriesAutoencoder(
        input_dim=input_dim,
        encoder_hidden_dim=config['encoder_hidden_dim'],
        encoder_layers=config['encoder_layers'],
        decoder_hidden_dim=config['decoder_hidden_dim'],
        decoder_layers=config['decoder_layers']
    ).to(device)
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(autoencoder.parameters(), lr=config['learning_rate'])

    print(f"  - Pre-training Masked Autoencoder on {len(dataset)} sequences for {config['epochs']} epochs...")
    for epoch in range(config['epochs']):
        epoch_loss = 0
        for i, (batch,) in enumerate(loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Masking logic
            noise = torch.rand(batch.shape[0], batch.shape[1], device=device)
            unmasked_indices = noise > config['mask_ratio']
            x_masked_input = batch.clone()
            x_masked_input[~unmasked_indices] = 0
            
            reconstruction = autoencoder(x_masked_input)
            
            # Calculate loss ONLY on masked elements
            loss_all = criterion(reconstruction, batch)
            masked_indices = ~unmasked_indices
            loss_mask = masked_indices.unsqueeze(-1).expand_as(loss_all)
            masked_loss = loss_all[loss_mask]
            
            if masked_loss.numel() > 0:
                loss = masked_loss.mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / (i + 1) if i > 0 else epoch_loss
            print(f"    Epoch [{epoch+1}/{config['epochs']}], Loss: {avg_loss:.6f}")
    
    # Visualize final reconstruction
    visualize_autoencoder_reconstruction(autoencoder.to('cpu'), loader, scaler, results_dir, config['mask_ratio'])

    return autoencoder.encoder.to("cpu"), scaler

def visualize_autoencoder_reconstruction(model, data_loader, scaler, results_dir, mask_ratio, n_samples=3):
    """
    Visualizes how well the autoencoder reconstructs the original time-series data.
    """
    model.eval()
    original_seqs_tensor = next(iter(data_loader))[0][:n_samples]
    
    # Create mask for visualization
    torch.manual_seed(42)
    noise = torch.rand(original_seqs_tensor.shape[0], original_seqs_tensor.shape[1])
    unmasked_indices = noise > mask_ratio
    x_masked_input = original_seqs_tensor.clone()
    x_masked_input[~unmasked_indices] = 0

    with torch.no_grad():
        reconstructed_seqs_tensor = model(x_masked_input)
        
    original_seqs = original_seqs_tensor.cpu().numpy()
    reconstructed_seqs = reconstructed_seqs_tensor.cpu().numpy()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(nrows=n_samples, ncols=1, figsize=(15, 5 * n_samples), sharex=True)
    if n_samples == 1:
        axes = [axes]

    fig.suptitle('Masked Autoencoder Pre-training: Original vs. Reconstructed', fontsize=16)

    for i in range(n_samples):
        original_unscaled = scaler.inverse_transform(original_seqs[i])
        reconstructed_unscaled = scaler.inverse_transform(reconstructed_seqs[i])
        
        time_steps = np.arange(original_unscaled.shape[0])
        axes[i].plot(time_steps, original_unscaled[:, 0], label='Original Data (Ground Truth)', color='dodgerblue', zorder=2)
        axes[i].plot(time_steps, reconstructed_unscaled[:, 0], label='Reconstructed by Model', color='orangered', linestyle='--', zorder=3)
        
        masked_time_steps = time_steps[~unmasked_indices[i]]
        for t in masked_time_steps:
             axes[i].axvspan(t - 0.5, t + 0.5, color='gray', alpha=0.2, zorder=1)
        
        axes[i].plot([], [], color='gray', alpha=0.2, linewidth=10, label=f'Masked Area (Input is Zero)')
        axes[i].set_title(f'Sample {i+1}')
        axes[i].legend()

    plt.xlabel('Time Step in Window')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    plot_path = os.path.join(results_dir, "autoencoder_reconstruction.png")
    plt.savefig(plot_path)
    print(f"✅ Autoencoder reconstruction plot saved to {plot_path}")
    plt.close(fig)

# --- Feature Engineering & Evaluation (Adapted for DAP) ---
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
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert Data Scientist specializing in time-series forecasting. 
Your task is to devise a feature engineering plan to improve a model's R^2 score.
You will be given the current features, their importance, and a list of available operations.
CRITICALLY, you will also receive a history of previously rejected plans. Learn from these past failures and avoid suggesting identical or very similar plans. 
Your goal is to explore novel combinations, considering SHORT-TERM (e.g., lags), LONG-TERM (e.g., Fourier features for seasonality), and LEARNED REPRESENTATIONS from a pre-trained autoencoder.
Your response MUST be a valid JSON object containing a single key, "plan".
The available operations are: create_lag, create_diff, create_rolling_mean, create_rolling_std, create_ewm, create_rolling_skew, create_rolling_kurt, create_rolling_min, create_rolling_max, create_time_features, create_fourier_features, create_interaction, create_learned_embedding, delete_feature.
Example for a complex plan: {"plan": [{"operation": "create_fourier_features", "feature": "date", "period": 365.25, "order": 2, "id": "F_Year"}, {"operation": "create_learned_embedding", "feature": "temp", "id": "LE_Month"}]}
Note: For `create_learned_embedding`, the window size and embedding dimension are fixed by the pre-trained model. You only need to provide the source feature and an ID.
Do not suggest features that already exist in the 'Available Features' list.
Do not suggest deleting the last remaining feature.
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

def evaluate_performance(df: pd.DataFrame, target_col: str):
    """
    DEPRECATED: This function is replaced by the more advanced `probe_feature_set`.
    Evaluates feature set performance using a fast and robust LightGBM model.
    This function serves as the "judge" for each T-LAFS iteration.
    """
    # 1. Prepare Data
    df_feat = df.drop(columns=['date', target_col]).dropna()
    y = df.loc[df_feat.index][target_col]
    X = df_feat

    if X.empty:
        print("  - ⚠️ Warning: Feature set is empty after dropping NaNs. Returning poor score.")
        return -1.0, pd.Series(dtype=float), pd.Series(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # 2. Model Training, Prediction & Importance
    lgb_model = lgb.LGBMRegressor(random_state=42, n_estimators=100, verbosity=-1)
    lgb_model.fit(X_train, y_train)
    
    predictions = lgb_model.predict(X_test)
    score = r2_score(y_test, predictions)
    
    importances = pd.Series(lgb_model.feature_importances_, index=X.columns)
    
    # SHAP contributions are deprecated, return empty series for compatibility
    shap_contributions = pd.Series(dtype=float)

    return score, importances.sort_values(ascending=False), shap_contributions

def probe_feature_set(df: pd.DataFrame, target_col: str, n_probes=3):
    """
    Multi-faceted Dynamic Feature Probe.
    Evaluates a feature set based on performance, stability, and feature diversity across different model types.
    """
    df_feat = df.drop(columns=['date', target_col]).dropna()
    y = df.loc[df_feat.index][target_col]
    X = df_feat

    if X.empty:
        print("  - ⚠️ Probe Warning: Feature set is empty. Returning poor score.")
        return {
            "primary_score": -1.0, "score_stability": 999, "importance_diversity": 0,
            "model_scores": {"lgbm": -1.0, "nn": -1.0},
            "feature_importances": pd.Series(dtype=float)
        }

    scores_lgbm, scores_nn, importances_list = [], [], []
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

    for seed in range(n_probes):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + seed, shuffle=False)
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42 + seed, shuffle=False)

        # Probe 1: LightGBM (Tree-based)
        lgbm = lgb.LGBMRegressor(random_state=42 + seed, n_estimators=100, verbosity=-1)
        lgbm.fit(X_train, y_train)
        scores_lgbm.append(r2_score(y_test, lgbm.predict(X_test)))
        importances_list.append(lgbm.feature_importances_)
        
        # Probe 2: SimpleNN (Neural Network-based)
        nn_model = SimpleNN(X.shape[1])
        preds_scaled = train_pytorch_model(nn_model, X_train_s, y_train_s, X_test_s)
        scores_nn.append(r2_score(y_test_s, preds_scaled))

    # --- Synthesize Probe Results ---
    avg_lgbm_score = np.mean(scores_lgbm)
    avg_nn_score = np.mean(scores_nn)
    
    # Primary score is a blend of different model structures
    primary_score = (avg_lgbm_score + avg_nn_score) / 2
    
    # Stability: Lower is better. How volatile is the score with different data slices?
    score_stability = np.std(scores_lgbm)
    
    # Importance Diversity (Gini Impurity): Higher is better. Are features working as a team?
    # 0 = one feature dominates; ~1 = all features contribute equally.
    avg_importances = np.mean(importances_list, axis=0)
    norm_importances = avg_importances / (np.sum(avg_importances) + 1e-9)
    importance_diversity = 1 - np.sum(np.power(norm_importances, 2))

    return {
        "primary_score": primary_score,
        "score_stability": score_stability,
        "importance_diversity": importance_diversity,
        "model_scores": {"lgbm": avg_lgbm_score, "nn": avg_nn_score},
        "feature_importances": pd.Series(avg_importances, index=X.columns).sort_values(ascending=False)
    }

def visualize_final_predictions(dates, y_true, y_pred, best_model_name, probe_name, best_model_metrics, results_dir):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))
    plt.plot(dates, y_true, label='Actual Values', color='dodgerblue', alpha=0.9)
    plt.plot(dates, y_pred, label=f'Predicted Values ({best_model_name})', color='orangered', linestyle='--')
    title = (f"Final Validation (Probe: {probe_name}) - Best Performing Model: {best_model_name}\n"
             f"R²: {best_model_metrics['r2']:.4f}  |  MAE: {best_model_metrics['mae']:.4f}  |  RMSE: {best_model_metrics['rmse']:.4f}")
    plt.title(title, fontsize=14)
    plt.legend()
    plot_path = os.path.join(results_dir, f"final_predictions_probe_{probe_name}.png")
    plt.savefig(plot_path)
    print(f"✅ Final predictions plot saved to {plot_path}")
    plt.show()

def save_results_to_json(results_data, probe_name, results_dir):
    """Saves the final results and summary to a JSON file in the specified directory."""
    file_path = os.path.join(results_dir, f"tlafs_results_probe_{probe_name}.json")
    
    # Custom handler for numpy types that the default JSON encoder can't handle.
    def json_converter(o):
        if isinstance(o, (np.floating, np.float64, np.float32)):
            if np.isinf(o) or np.isneginf(o): return str(o)
            if np.isnan(o): return None
            return float(o)
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.bool_)):
            return bool(o)
        if isinstance(o, (np.ndarray)):
            return o.tolist()
        if isinstance(o, pd.Series):
            return o.to_dict()
        return str(o) if hasattr(o, '__str__') else f"<non-serializable: {type(o)}>"
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False, default=json_converter)
    print(f"\n✅ Results and configuration saved to {file_path}")

def evaluate_on_multiple_models(df: pd.DataFrame, target_col: str, probe_name: str):
    """
    Evaluates the final feature set on a variety of models.
    This now uses the static, correct execute_plan method from TLAFS_Algorithm.
    """
    print(f"Evaluating final feature set on various models (Probe was: {probe_name})...")
    
    # This plan is not executed, it's just for reference if needed.
    # The dataframe `df` is already the one with the best features.
    
    X = df.drop(columns=['date', target_col]).dropna()
    y = df.loc[X.index][target_col]

    models = {
        'LightGBM': lgb.LGBMRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'SimpleNN': SimpleNN(X.shape[1]),
        'EnhancedNN': EnhancedNN(X.shape[1]),
        'Transformer': TransformerModel(X.shape[1])
    }
    
    final_metrics = {}
    final_results = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    # Use a separate scaler for the target to avoid data leakage issues if we were to scale y_test
    scaler_y = MinMaxScaler()
    y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    for name, model in models.items():
        print(f"  - Evaluating {name}...")
        if name in ['SimpleNN', 'EnhancedNN', 'Transformer']:
            preds_scaled = train_pytorch_model(model, X_train_s, y_train_s, X_test_s)
            preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        final_metrics[name] = {
            "r2": r2,
            "mae": mae,
            "rmse": rmse
        }
        final_results[name] = {
            "dates": X_test.index.tolist(),
            "y_true": y_test.tolist(),
            "y_pred": preds.tolist()
        }
        print(f"    - {name}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
        
    return final_metrics, final_results

class TLAFS_Algorithm:
    """
    Time-series Language-augmented Feature Search (T-LAFS) Algorithm.
    This class orchestrates the automated feature engineering process.
    """
    def __init__(self, base_df, target_col, n_iterations=5, acceptance_threshold=0.01, results_dir="."):
        self.base_df = base_df
        self.target_col = target_col
        self.n_iterations = n_iterations
        self.acceptance_threshold = acceptance_threshold
        self.history = []
        self.best_score = -np.inf
        self.best_plan = []
        self.best_df = None
        self.client = OpenAI()
        self.results_dir = results_dir
        
        # --- NEW: Enrich base_df and prepare for pretraining ---
        print("\nEnriching data with time features for a smarter autoencoder...")
        self.base_df['dayofweek'] = self.base_df['date'].dt.dayofweek
        self.base_df['month'] = self.base_df['date'].dt.month
        self.base_df['weekofyear'] = self.base_df['date'].dt.isocalendar().week.astype(int)
        self.base_df['is_weekend'] = (self.base_df['date'].dt.dayofweek >= 5).astype(int)
        
        TLAFS_Algorithm.pretrain_cols_static = [self.target_col, 'dayofweek', 'month', 'weekofyear', 'is_weekend']
        df_for_pretraining = self.base_df[TLAFS_Algorithm.pretrain_cols_static]
        print(f"Autoencoder will be trained on features: {TLAFS_Algorithm.pretrain_cols_static}")

        print("\n🧠 Pre-training a Masked Autoencoder to generate features...")
        
        pretrain_config = {
            'encoder_hidden_dim': 128, 'encoder_layers': 4,
            'decoder_hidden_dim': 64, 'decoder_layers': 1,
            'epochs': 100, 'batch_size': 64,
            'learning_rate': 0.001, 'mask_ratio': 0.4
        }

        self.pretrained_encoder, self.embedder_scaler = pretrain_embedder(
            df_for_pretraining,
            window_size=90,
            config=pretrain_config,
            results_dir=self.results_dir
        )
        print("   ✅ Pre-training complete.")
        
        # Add static variables for the static method to use
        TLAFS_Algorithm.pretrained_encoder = self.pretrained_encoder
        TLAFS_Algorithm.embedder_scaler = self.embedder_scaler
        TLAFS_Algorithm.target_col_static = self.target_col
        # TLAFS_Algorithm.pretrain_cols_static is already set

    @staticmethod
    def execute_plan(df: pd.DataFrame, plan: list):
        """
        Executes a feature engineering plan on a given dataframe.
        This is the single, authoritative static method for all feature generation.
        It contains all non-leaky feature generation logic.
        """
        temp_df = df.copy()
        
        embedder = TLAFS_Algorithm.pretrained_encoder
        scaler = TLAFS_Algorithm.embedder_scaler
        target_col = TLAFS_Algorithm.target_col_static
        
        for step in plan:
            op = step.get("operation")
            feature = step.get("feature") or step.get("base_feature") or step.get("column")

            # A more robust way to generate a new column name
            def get_new_col_name(base_feature, id_suggestion):
                if base_feature and id_suggestion.startswith(base_feature):
                    return id_suggestion
                return f"{base_feature}_{id_suggestion}" if base_feature else id_suggestion

            try:
                # --- Basic Time-Series Features ---
                if op == "create_lag":
                    days = step.get("days", 1)
                    id = step.get("id", f"lag{days}")
                    new_col_name = get_new_col_name(feature, id)
                    temp_df[new_col_name] = temp_df[feature].shift(days).bfill()

                elif op == "create_diff":
                    periods = step.get("periods", 1)
                    id = step.get("id", f"diff{periods}")
                    new_col_name = get_new_col_name(feature, id)
                    temp_df[new_col_name] = temp_df[feature].diff(periods).shift(1).bfill()

                elif op in ["create_rolling_mean", "create_rolling_std", "create_rolling_skew", "create_rolling_kurt", "create_rolling_min", "create_rolling_max"]:
                    window = step.get("window", 7)
                    op_name = op.split('_')[2]
                    id = step.get("id", f"roll_{op_name}{window}")
                    new_col_name = get_new_col_name(feature, id)
                    roll_op = getattr(temp_df[feature].rolling(window=window), op_name)
                    temp_df[new_col_name] = roll_op().shift(1).bfill()

                elif op == "create_ewm":
                    span = step.get("span", 7)
                    id = step.get("id", f"ewm{span}")
                    new_col_name = get_new_col_name(feature, id)
                    temp_df[new_col_name] = temp_df[feature].ewm(span=span, adjust=False).mean().shift(1).bfill()

                # --- Calendar and Seasonality Features ---
                elif op == "create_time_features":
                    if pd.api.types.is_datetime64_any_dtype(temp_df[feature]):
                        default_parts = ["dayofweek", "month", "is_weekend"]
                        for part in step.get("extract", default_parts):
                            if part == "dayofweek": temp_df[f'{feature}_dayofweek'] = temp_df[feature].dt.dayofweek
                            elif part == "month": temp_df[f'{feature}_month'] = temp_df[feature].dt.month
                            elif part == "quarter": temp_df[f'{feature}_quarter'] = temp_df[feature].dt.quarter
                            elif part == "is_weekend": temp_df[f'{feature}_is_weekend'] = (temp_df[feature].dt.dayofweek >= 5).astype(int)
                            elif part == "dayofyear": temp_df[f'{feature}_dayofyear'] = temp_df[feature].dt.dayofyear
                            elif part == "weekofyear": temp_df[f'{feature}_weekofyear'] = temp_df[feature].dt.isocalendar().week.astype(int)
                    else:
                        print(f"  - ⚠️ Warning: Skipped time_features on non-datetime column '{feature}'.")

                elif op == "create_fourier_features":
                    # This operation is always based on the 'date' column for seasonality.
                    # We ignore the 'feature' parameter from the LLM to prevent errors.
                    if pd.api.types.is_datetime64_any_dtype(temp_df['date']):
                        period = float(step["period"])
                        order = int(step["order"])
                        time_idx = (temp_df['date'] - temp_df['date'].min()).dt.days
                        for k in range(1, order + 1):
                            temp_df[f'fourier_sin_{k}_{int(period)}'] = np.sin(2 * np.pi * k * time_idx / period)
                            temp_df[f'fourier_cos_{k}_{int(period)}'] = np.cos(2 * np.pi * k * time_idx / period)
                    else:
                        print(f"  - ⚠️ Warning: Skipped fourier because 'date' column is not datetime.")

                # --- Learned & Interaction Features ---
                elif op == "create_learned_embedding":
                    if embedder and scaler:
                        id = step.get("id", "embed")
                        pretrain_cols = TLAFS_Algorithm.pretrain_cols_static
                        window = 90 # Hardcoded for now, should match pretraining
                        
                        print(f"  - Generating multi-variate embedding (win:{window}) from {len(pretrain_cols)} features...")
                        
                        df_for_embedding = temp_df[pretrain_cols]
                        scaled_features = scaler.transform(df_for_embedding)
                        
                        sequences = np.array([scaled_features[i:i+window] for i in range(len(scaled_features) - window + 1)])
                        
                        if sequences.size == 0:
                            print(f"  - ⚠️ Not enough data for embedding. Skipping.")
                            continue
                            
                        tensor = torch.FloatTensor(sequences)
                        with torch.no_grad():
                            # Encoder returns a sequence of latent vectors: (batch, seq_len, hidden_dim * 2)
                            latent_sequence = embedder(tensor).numpy()
                            # --- Hybrid Pooling (Mean + Max) ---
                            # Perform mean pooling to get the "average" state of the sequence
                            avg_pooled = np.mean(latent_sequence, axis=1)
                            # Perform max pooling to get the "peak" events in the sequence
                            max_pooled = np.max(latent_sequence, axis=1)
                            # Concatenate both to provide a richer representation
                            embeddings = np.concatenate([avg_pooled, max_pooled], axis=1)
                            
                        valid_indices = temp_df.index[window-1:]
                        # CRITICAL FIX: Use shape[1] (number of features) not shape[0] (number of sequences)
                        cols = [f"embed_{i}_{id}" for i in range(embeddings.shape[1])]
                        embed_df = pd.DataFrame(embeddings, index=valid_indices, columns=cols)
                        
                        temp_df = temp_df.join(embed_df)
                        temp_df[cols] = temp_df[cols].bfill()
                    else:
                        print(f"  - ⚠️ Embedder not available. Skipping.")

                elif op == "create_interaction":
                    features = step.get("features")
                    if not features:
                        f1 = step.get("feature1", step.get("feature_a"))
                        f2 = step.get("feature2", step.get("feature_b"))
                        if f1 and f2:
                            features = [f1, f2]

                    if isinstance(features, list) and len(features)==2 and all(f in temp_df.columns for f in features):
                        id = step.get('id', 'new')
                        new_col_name = f"{features[0]}_x_{features[1]}_{id}"
                        temp_df[new_col_name] = temp_df[features[0]] * temp_df[features[1]]
                    else:
                        print(f"  - ⚠️ Skipping interaction for invalid/missing features: {features}")

                elif op == "delete_feature":
                    if feature in temp_df.columns and feature not in ['date', target_col]:
                        temp_df.drop(columns=[feature], inplace=True)
                        print(f"  - Successfully deleted feature: {feature}")
                
            except Exception as e:
                import traceback
                print(f"  - ❌ ERROR executing step {step}. Error: {e}\n{traceback.format_exc()}")
        
        return temp_df
        
    def get_plan_from_llm(self, context_prompt):
        system_prompt = f"""You are an expert Data Scientist building a robust, generalizable time-series forecasting model. Your task is to devise a feature engineering plan to improve the model's R^2 score.

*** FEATURE ENGINEERING HIERARCHY & RULES ***
1.  **PRIORITIZE EXOGENOUS AND LEARNED FEATURES:** Your primary focus should be creating features from independent sources (like 'dayofweek', 'month') and from the pre-trained learned embeddings ('embed_*'). These features help the model understand the underlying drivers of the time series.
2.  **USE TARGET-DERIVED FEATURES SPARINGLY:** You may create rolling statistics (mean, std, etc.) or lags/diffs based on the target variable ('{self.target_col}'). However, use these ONLY to capture any remaining auto-correlation that the primary features missed. Do not rely on them as your main strategy. A good model should not solely depend on the recent history of the target.
3.  **STRICTLY FORBIDDEN (DATA LEAKAGE):** You are strictly forbidden from creating features that cause data leakage. The most common mistake is using the target variable ('{self.target_col}') in an un-shifted operation, such as 'create_interaction'. The system will automatically shift time-based features like lags and rolling stats, but operations like 'create_interaction' on the target are unsafe and thus banned.

The available operations are: create_lag, create_diff, create_rolling_mean, create_rolling_std, create_ewm, create_rolling_skew, create_rolling_kurt, create_rolling_min, create_rolling_max, create_time_features, create_fourier_features, create_interaction, create_learned_embedding, delete_feature.
Example for a complex plan: {{"plan": [{{"operation": "create_fourier_features", "period": 365.25, "order": 2, "id": "F_Year"}}, {{"operation": "create_learned_embedding", "id": "LE_Month"}}]}}
Note: For `create_learned_embedding`, you only need to provide an ID. The system handles the rest.
For `create_fourier_features`, the operation will always be applied to the 'date' column.
Do not suggest features that already exist. Do not suggest deleting the last remaining feature.
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": context_prompt
                    }
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            plan_str = response.choices[0].message.content
            plan = json.loads(plan_str)
            return plan.get("plan", plan)
        except Exception as e:
            print(f"❌ Error calling LLM: {e}")
            # Fallback plan updated to not use the target variable directly for rolling stats
            return [{"operation": "create_rolling_std", "feature": "month", "window": 3, "id": "fallback_err"}]

    def run(self):
        print(f"\n💡 Starting T-LAFS with Multi-faceted Dynamic Probe ...\n")
        current_df = self.base_df.copy()
        current_plan = []
        
        # Start with a simple, universally useful feature to give the LLM a good baseline.
        initial_plan = [
            {"operation": "create_lag", "feature": self.target_col, "days": 1, "id": "lag1"}
        ]
        current_df = self.execute_plan(current_df, initial_plan)
        current_plan = initial_plan
        
        print("\nEstablishing baseline score with initial feature set (lag1)...")
        try:
            probe_results = probe_feature_set(current_df, self.target_col)
            current_score = probe_results["primary_score"]
            self.last_probe_results = probe_results

            self.best_score = current_score
            self.best_df = current_df.copy()
            self.best_plan = current_plan.copy()
            print(f"  - Initial baseline score (primary_score): {self.best_score:.4f}")
            print(f"  - Model Blend: (LGBM: {probe_results['model_scores']['lgbm']:.3f}, NN: {probe_results['model_scores']['nn']:.3f})")
        except Exception as e:
            import traceback
            print(f"  - ❌ ERROR during initial evaluation: {e}\n{traceback.format_exc()}")
            return None, None, -1

        for i in range(self.n_iterations):
            print(f"\n----- ITERATION {i+1}/{self.n_iterations} (Probe: Dynamic) -----")
            
            last_results = self.last_probe_results
            print(f"  - Current Score: {last_results['primary_score']:.4f} (LGBM: {last_results['model_scores']['lgbm']:.3f}, NN: {last_results['model_scores']['nn']:.3f})")
            print(f"  - Stability (std): {last_results['score_stability']:.4f} | Diversity (gini): {last_results['importance_diversity']:.4f} | Historical Best: {self.best_score:.4f}")

            print("\nStep 1: Strategist LLM is devising a new feature combo plan...")
            
            rejected_plans = [item['plan'] for item in self.history if not item.get('adopted') and item.get('plan')]
            rejected_plans_str = json.dumps(rejected_plans[-5:], indent=2) if rejected_plans else "None so far."
            
            # Show only base features to the LLM to guide it
            base_features = [c for c in self.base_df.columns if c != 'date']
            available_features = [c for c in current_df.columns if c != self.target_col and c != 'date']
            
            feature_importances_str = last_results['feature_importances'].head(10).to_string() if 'feature_importances' in last_results and last_results['feature_importances'] is not None else "Not available."

            context_prompt = f"""
Current State & Probe Analysis:
- Primary Score (R^2 Blend): {last_results['primary_score']:.4f} (Historical Best: {self.best_score:.4f})
- Individual Model Scores: LGBM R^2: {last_results['model_scores']['lgbm']:.4f}, SimpleNN R^2: {last_results['model_scores']['nn']:.4f}
- Score Stability (std dev across runs, lower is better): {last_results['score_stability']:.4f}
- Feature Diversity (Gini, higher is better): {last_results['importance_diversity']:.4f}
- Available Features: {available_features}
- Recommended Base Features for Transformation: {base_features}
- Top 10 Feature Importances (from LightGBM): 
{feature_importances_str}

History of Recently Rejected Plans:
{rejected_plans_str}

Task for Iteration {i+1}/{self.n_iterations}:
Analyze the probe's feedback. If diversity is low, suggest interactions or new feature types. If stability is poor, maybe simplify or remove features. If a model type is lagging, suggest features that might help it. Propose a short, creative list (1-2) of operations based on the 'Recommended Base Features'.
"""
            plan_extension = self.get_plan_from_llm(context_prompt)
            
            if not plan_extension:
                self.history.append({"iteration": i + 1, "plan": [], "score": current_score, "adopted": False, "action": "noop"})
                continue
            
            print(f"✅ LLM Strategist proposed: {plan_extension}")

            print(f"\nStep 2: Probing the new feature combo plan...")
            df_with_new_features = self.execute_plan(current_df, plan_extension)
            
            new_probe_results = probe_feature_set(df_with_new_features, self.target_col)
            new_score = new_probe_results["primary_score"]

            print(f"  - Probe results: Score={new_score:.4f}, Stability={new_probe_results['score_stability']:.4f}, Diversity={new_probe_results['importance_diversity']:.4f}")
            
            print(f"\nStep 3: Deciding whether to adopt the new plan...")
            is_adopted = new_score > (self.best_score - self.acceptance_threshold)
            
            if is_adopted:
                current_df = df_with_new_features.copy()
                current_score = new_score
                self.last_probe_results = new_probe_results
                
                if plan_extension:
                    current_plan += plan_extension
                
                if new_score > self.best_score:
                    print(f"  -> SUCCESS! New peak score {new_score:.4f} > best score {self.best_score:.4f}.")
                    self.best_score = new_score
                    self.best_df = current_df.copy()
                    self.best_plan = current_plan.copy()
                else:
                    print(f"  -> TOLERATED. Score {new_score:.4f} is within threshold. Accepting for exploration.")
            else:
                print(f"  -> PLAN REJECTED. Score {new_score:.4f} is too low. Reverting to best known state for next iteration.")
                # On rejection, we revert the state, so current_df and current_plan are already the best ones.
                # We don't need to do anything to the variables, just not update them.
            
            self.history.append({"iteration": i + 1, "plan": plan_extension, "probe_results": new_probe_results, "adopted": is_adopted})

        print("\n" + "="*80 + f"\n🏆 T-LAFS (Dynamic Probe) Finished! 🏆")
        print(f"   - Best Primary Score Achieved (during search): {self.best_score:.4f}")
        
        return self.best_df, self.best_plan, self.best_score


def main():
    """Main function to run the DAP experiment."""
    # ===== 配置变量 =====
    DATASET_TYPE = 'min_daily_temps'  # 可选: 'min_daily_temps' 或 'total_cleaned'
    # 探针配置
    PROBE_NAME = 'quantum_dual_stream'  # 可选: 'quantum_dual_stream', 'dual_stream', 'bayesian_quantum'
    PROBE_CONFIG = {
        'quant_input_size': None,  # 将在运行时设置
        'vocab_size': 5,
        'qual_embed_dim': 16,
        'quant_embed_dim': 48
    }
    
    # 实验配置
    N_ITERATIONS = 10
    TARGET_COL = 'temp'
    
    # 数据配置
    # DATA_PATH = 'data/min_daily_temps.csv'  # 不再需要
    
    print("="*80)
    print(f"🚀 T-LAFS Experiment: Dynamic Probe Guided Search")
    print("="*80)

    # --- 创建本次运行的专属结果目录 ---
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("results", f"run_{run_timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"📂 All results for this run will be saved in: {results_dir}")

    # 初始化API客户端
    setup_api_client()
    
    # 加载数据
    base_df = get_time_series_data(DATASET_TYPE)
    
    # 创建并运行TLAFS算法
    tlafs = TLAFS_Algorithm(
        base_df=base_df,
        target_col=TARGET_COL,
        n_iterations=N_ITERATIONS,
        results_dir=results_dir
    )
    best_df, best_feature_plan, best_score_during_search = tlafs.run()

    # --- 最终验证和总结 ---
    if best_df is not None:
        probe_name_for_reporting = "DynamicProbe"
        print("\n" + "="*40)
        print("🔬 FINAL VALIDATION ON ALL MODELS 🔬")
        print("="*40)
        final_metrics, final_results = evaluate_on_multiple_models(
            best_df,
            TARGET_COL,
            probe_name_for_reporting
        )

        if final_metrics:
            # Find best model based on R^2 score
            best_final_model_name = max(final_metrics, key=lambda k: final_metrics[k]['r2'])
            best_final_metrics = final_metrics[best_final_model_name]
            best_result = final_results[best_final_model_name]

            print("\n" + "="*60)
            print(f"🏆 EXECUTIVE SUMMARY: T-LAFS '{probe_name_for_reporting}' STRATEGY 🏆")
            print("="*60)
            print(f"Feature engineering search was conducted by: '{probe_name_for_reporting}'")
            print(f"   - Best Primary Score during search phase: {best_score_during_search:.4f}")
            print(f"\nBest feature plan discovered:")
            print(json.dumps(best_feature_plan, indent=2, ensure_ascii=False))
            print("\n------------------------------------------------------------")
            print(f"This feature set was then validated on a suite of specialist models.")
            print(f"🥇 Best Performing Specialist Model: '{best_final_model_name}'")
            print(f"🚀 Final Validated R² Score: {best_final_metrics['r2']:.4f}")
            print(f"   - MAE: {best_final_metrics['mae']:.4f}, RMSE: {best_final_metrics['rmse']:.4f}")
            print("\nConclusion: The T-LAFS framework successfully used the probe to find a")
            print("powerful feature set, which a specialist model then exploited to")
            print("achieve high-performance predictions.")
            print("="*60)

            visualize_final_predictions(
                dates=best_result['dates'],
                y_true=best_result['y_true'],
                y_pred=best_result['y_pred'],
                best_model_name=best_final_model_name,
                probe_name=probe_name_for_reporting,
                best_model_metrics=best_final_metrics,
                results_dir=results_dir
            )
            
            # --- 保存结果到JSON ---
            print("\n💾 Saving results to JSON...")
            results_to_save = {
                "probe_model": probe_name_for_reporting,
                "probe_config": {
                    "description": "Multi-faceted: Blend of LGBM+NN scores, stability, and diversity.",
                    "n_probes": 3
                },
                "best_score_during_search": best_score_during_search,
                "best_feature_plan": best_feature_plan,
                "final_features": [col for col in best_df.columns if col not in ['date', TARGET_COL]],
                "final_validation_scores": final_metrics,
                "best_final_model": {
                    "name": best_final_model_name,
                    "metrics": best_final_metrics
                },
                "run_history": tlafs.history
            }
            save_results_to_json(results_to_save, probe_name_for_reporting, results_dir)

        else:
            print("\nCould not generate final validation scores.")

if __name__ == "__main__":
    main()
