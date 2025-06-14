import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import math

# --- Model Definitions (NEW: Agent Attention Probe Architecture) ---

class AgentAttentionProbe(nn.Module):
    """
    A module that uses a set of 'agent' queries to probe a time series
    and distill its information into a fixed-size embedding.
    """
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_agents: int):
        super().__init__()
        self.num_agents = num_agents
        # The 'probes' or 'agents' that will query the sequence
        self.agents = nn.Parameter(torch.randn(1, num_agents, d_model))
        
        # A standard transformer encoder layer to process the sequence
        # We need to project the input into d_model dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        # CORRECTED: Add positional encoding here, where the dimension is d_model
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # The cross-attention mechanism where agents query the encoded sequence
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # 1. Project input to the model's dimension
        x_proj = self.input_proj(x) # -> (batch_size, seq_len, d_model)

        # 2. Add positional encoding to the projected sequence
        x_pos = self.pos_encoder(x_proj)
        
        # 3. Encode the sequence
        encoded_seq = self.transformer_encoder(x_pos) # -> (batch_size, seq_len, d_model)

        # 4. Agents query the encoded sequence
        # We need to expand agents for each item in the batch
        batch_size = x.shape[0]
        agent_queries = self.agents.expand(batch_size, -1, -1) # -> (batch_size, num_agents, d_model)

        # The agents are the queries, the encoded sequence is the key and value
        attn_output, _ = self.cross_attention(query=agent_queries, key=encoded_seq, value=encoded_seq)
        # attn_output shape: (batch_size, num_agents, d_model)
        
        # --- MODIFIED: Return the full output of all agents, not the average ---
        # The flattening and processing will be handled by the forecaster
        return attn_output

class ProbeForecaster(nn.Module):
    """
    Combines a global embedding from the probe with local features for forecasting.
    - Global feature: A learned embedding summarizing the entire input sequence.
    - Local feature: Raw values from the most recent N time steps (lags).
    - Exogenous feature: Future known values (e.g., day of week for the day we are predicting).
    """
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_agents: int, num_lags: int, num_exog: int, dropout: float = 0.1):
        super().__init__()
        # The probe now handles its own positional encoding
        self.probe = AgentAttentionProbe(input_dim, d_model, nhead, num_agents)
        
        self.num_lags = num_lags
        
        # --- MODIFIED: Calculate input size for the head based on the full probe output ---
        # The probe now outputs (num_agents, d_model) for each sample.
        probe_feature_size = num_agents * d_model
        
        # Calculate the size of the flattened local features
        # Note: local features include all input dims (temp, exog)
        local_feature_size = num_lags * input_dim

        # Calculate the total input size for the forecasting head
        forecasting_input_size = probe_feature_size + local_feature_size + num_exog

        self.forecasting_head = nn.Sequential(
            nn.Linear(forecasting_input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1) 
        )

    def forward(self, x, x_exog):
        # x shape: (batch_size, seq_len, input_dim)
        # x_exog shape: (batch_size, num_exog)
        
        # 1. Get the full agent output from the probe. 
        # Shape: (batch_size, num_agents, d_model)
        probe_output = self.probe(x)
        
        # --- MODIFIED: Flatten the probe's output to use all agent information ---
        probe_output_flat = probe_output.reshape(probe_output.shape[0], -1)

        # 2. Get local features (e.g., last N lags) from the original sequence
        # Note: we use all input features for the lags
        # Ensure we don't go out of bounds if seq_len < num_lags
        actual_lags = min(self.num_lags, x.shape[1])
        # Pad with zeros if sequence is shorter than num_lags
        if actual_lags < self.num_lags:
            padding = torch.zeros(x.shape[0], self.num_lags - actual_lags, x.shape[2]).to(x.device)
            local_lags = torch.cat([padding, x], dim=1)
        else:
            local_lags = x[:, -self.num_lags:, :] 

        # -> (batch_size, num_lags * input_dim)
        local_lags_flat = local_lags.reshape(x.shape[0], -1)

        # 3. Combine global, local, and future exogenous features
        combined_features = torch.cat((probe_output_flat, local_lags_flat, x_exog), dim=1)

        # 4. Make the prediction
        prediction = self.forecasting_head(combined_features)
        # --- MODIFIED: Return the flattened probe output for potential analysis ---
        return prediction, probe_output_flat

class PositionalEncoding(nn.Module):
    """Adds positional information to the input embeddings."""
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

# --- Data & Training Functions ---

def get_data():
    """Loads and preprocesses the time-series data."""
    df = pd.read_csv('data/total_cleaned.csv')
    df.rename(columns={'日期': 'date', '成交商品件数': 'temp'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    # --- UPDATED: Return all relevant features ---
    return df[['temp', 'dayofweek', 'month']]

def create_sequences_with_exog(data, seq_len):
    """
    Creates overlapping sequences (X, y) for forecasting.
    X includes all features, y is only the target variable.
    Also returns exogenous features for the forecast step.
    """
    xs, ys_target, ys_exog = [], [], []
    # data is now a numpy array
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len), :]
        y_target = data[i + seq_len, 0:1] # Target is the first column ('temp')
        y_exog = data[i + seq_len, 1:]   # Exogenous features are the rest
        xs.append(x)
        ys_target.append(y_target)
        ys_exog.append(y_exog)
    return np.array(xs), np.array(ys_target), np.array(ys_exog)

def train_forecaster(config):
    """Main training loop for the supervised forecaster."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and prepare data
    df = get_data()

    # --- Create Training and Validation Sets (CORRECTED: Split before scaling) ---
    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx]
    val_df = df[split_idx:]

    # --- CORRECTED: Use separate scalers for target and exogenous features ---
    target_scaler = MinMaxScaler()
    exog_scaler = MinMaxScaler()

    # Fit scalers ONLY on training data
    # Reshape is needed for single-column data
    train_target_scaled = target_scaler.fit_transform(train_df[['temp']])
    train_exog_scaled = exog_scaler.fit_transform(train_df[['dayofweek', 'month']])
    
    # Combine scaled data for sequence creation
    train_data_scaled = np.concatenate([train_target_scaled, train_exog_scaled], axis=1)

    # Use the same scalers to transform validation data
    val_target_scaled = target_scaler.transform(val_df[['temp']])
    val_exog_scaled = exog_scaler.transform(val_df[['dayofweek', 'month']])
    val_data_scaled = np.concatenate([val_target_scaled, val_exog_scaled], axis=1)
    
    # Create sequences for forecasting from scaled data
    X_train, y_train_target, y_train_exog = create_sequences_with_exog(train_data_scaled, config['seq_len'])
    X_val, y_val_target, y_val_exog = create_sequences_with_exog(val_data_scaled, config['seq_len'])

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train_exog), torch.FloatTensor(y_train_target))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val_exog), torch.FloatTensor(y_val_target))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 2. Initialize model and optimizer
    # input_dim is the number of features in the sequence
    input_dim = X_train.shape[2] 
    # num_exog is the number of future exogenous features
    num_exog = y_train_exog.shape[1]

    model = ProbeForecaster(
        input_dim=input_dim,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_agents=config['num_agents'],
        num_lags=config['num_lags'],
        num_exog=num_exog,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss() 

    # 3. Training loop with Early Stopping
    print("Starting forecaster training...")
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for i, (inputs, exog_inputs, targets) in enumerate(train_loader):
            inputs, exog_inputs, targets = inputs.to(device), exog_inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            predictions, _ = model(inputs, exog_inputs)
            loss = criterion(predictions, targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / (i + 1)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (inputs, exog_inputs, targets) in enumerate(val_loader):
                inputs, exog_inputs, targets = inputs.to(device), exog_inputs.to(device), targets.to(device)
                predictions, _ = model(inputs, exog_inputs)
                loss = criterion(predictions, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / (i + 1)

        print(f"Epoch [{epoch+1}/{config['epochs']}] | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f}")

        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            print(f"  -> New best validation loss: {best_val_loss:.6f}. Saving model.")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= config['patience']:
            print(f"\nEarly stopping triggered after {config['patience']} epochs with no improvement.")
            break
            
    print("Training finished.")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model state from early stopping.")

    # CORRECTED: Return target_scaler for proper inverse transformation
    return model.to('cpu'), target_scaler, val_loader # Return val_loader for visualization

def visualize_predictions(model, target_scaler, loader, config, n_samples=20):
    """
    Visualizes model predictions against actual values.
    CORRECTED: Uses the dedicated target_scaler for inverse transform.
    """
    print(f"\n--- Generating forecast visualization for seq_len={config['seq_len']} ---")
    model.eval()
    
    all_preds = []
    all_targets = []
    # The loader now yields batches of (inputs, exog_inputs, targets)
    with torch.no_grad():
        for inputs, exog_inputs, targets in loader:
            predictions, _ = model(inputs, exog_inputs)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # --- CORRECTED: Inverse transform using the dedicated target_scaler ---
    # No need for dummy arrays anymore.
    actual_values = target_scaler.inverse_transform(all_targets)
    predicted_values = target_scaler.inverse_transform(all_preds)
    
    # Squeeze the arrays to be 1D for metrics and plotting
    actual_values = actual_values.squeeze()
    predicted_values = predicted_values.squeeze()
    
    # Calculate metrics over the ENTIRE validation set
    r2 = r2_score(actual_values, predicted_values)
    mae = mean_absolute_error(actual_values, predicted_values)

    # Plotting (show a limited number of samples for clarity)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))
    
    plot_actuals = actual_values[:n_samples]
    plot_preds = predicted_values[:n_samples]

    time_steps = np.arange(len(plot_actuals))
    plt.plot(time_steps, plot_actuals, 'o-', label='Actual Values', color='dodgerblue')
    plt.plot(time_steps, plot_preds, 'o--', label='Predicted Values', color='orangered')
    
    plt.title(f"Forecaster Performance on Validation Set\n$R^2 = {r2:.3f}$ | MAE = {mae:.2f}", fontsize=16)
    plt.xlabel(f"Sample Index from Validation Set (showing first {n_samples})")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/forecaster_performance_len{config['seq_len']}.png"
    plt.savefig(plot_path)
    print(f"\n✅ Forecast plot saved to {plot_path}")
    plt.show()

def main():
    config = {
        'seq_len': 365,         # Capture full annual seasonality
        'd_model': 64,
        'nhead': 4,             
        'num_agents': 8,        
        'num_lags': 14,         # Use last 14 days as direct local features
        'epochs': 150,          # More epochs for more complex data
        'patience': 25,         # More patience
        'batch_size': 32,
        'learning_rate': 0.0005
    }
    
    print("\n" + "="*80)
    print(f"🚀 STARTING HYBRID PROBE EXPERIMENT (Window: {config['seq_len']}, Agents: {config['num_agents']}, Lags: {config['num_lags']}, Exog: True)")
    print("="*80)
    
    trained_model, target_scaler, loader = train_forecaster(config)
    visualize_predictions(trained_model, target_scaler, loader, config)

if __name__ == "__main__":
    main() 