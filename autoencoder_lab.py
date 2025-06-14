import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os

# --- Model Definitions (Bottleneck Architecture) ---

class MaskedEncoder(nn.Module):
    """
    A powerful, deep encoder that processes a sequence, passes it through a bottleneck,
    and outputs a single, low-dimensional latent vector representing the entire sequence.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, final_embedding_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # Bidirectional GRU output is 2 * hidden_dim
            nn.ReLU(),
            nn.Linear(hidden_dim, final_embedding_dim) # Project to final low-dimensional embedding
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        _, h_n = self.gru(x) # h_n shape: (num_layers * 2, batch_size, hidden_dim)
        
        # Concatenate the final forward and backward hidden states from the last layer
        last_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) # shape: (batch_size, hidden_dim * 2)
        
        embedding = self.projection(last_hidden) # shape: (batch_size, final_embedding_dim)
        return embedding

class MaskedDecoder(nn.Module):
    """A powerful decoder that takes a single latent vector and reconstructs the original input sequence."""
    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int, seq_len: int, num_layers: int):
        super().__init__()
        self.seq_len = seq_len
        self.expansion_fc = nn.Linear(embedding_dim, hidden_dim * 2)
        self.gru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x shape: (batch_size, embedding_dim)
        x_expanded = self.expansion_fc(x)
        x_repeated = x_expanded.unsqueeze(1).repeat(1, self.seq_len, 1)
        outputs, _ = self.gru(x_repeated)
        reconstruction = self.fc(outputs)
        return reconstruction

class MaskedTimeSeriesAutoencoder(nn.Module):
    """Combines the new Encoder and Decoder for pre-training with a bottleneck."""
    def __init__(self, input_dim: int, encoder_hidden_dim: int, encoder_layers: int, decoder_hidden_dim: int, decoder_layers: int, final_embedding_dim: int, seq_len: int):
        super().__init__()
        self.encoder = MaskedEncoder(
            input_dim=input_dim, 
            hidden_dim=encoder_hidden_dim, 
            num_layers=encoder_layers,
            final_embedding_dim=final_embedding_dim
        )
        self.decoder = MaskedDecoder(
            embedding_dim=final_embedding_dim, 
            hidden_dim=decoder_hidden_dim, 
            output_dim=input_dim, 
            seq_len=seq_len,
            num_layers=decoder_layers
        )

    def forward(self, x_masked):
        latent_embedding = self.encoder(x_masked)
        reconstruction = self.decoder(latent_embedding)
        return reconstruction

# --- Data & Training Functions ---

def get_data():
    """Loads and preprocesses the time-series data."""
    df = pd.read_csv('data/min_daily_temps.csv')
    df.rename(columns={'Date': 'date', 'Temp': 'temp'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df[['temp', 'dayofweek', 'month']]

def create_sequences(data, seq_len):
    """Creates overlapping sequences from the time-series data."""
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)

def train_masked_autoencoder(config):
    """Main training loop for the masked autoencoder, now with validation and early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and prepare data
    df = get_data()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    
    sequences = create_sequences(data_scaled, config['seq_len'])
    
    # --- Create Training and Validation Sets ---
    # For time series, we should not shuffle. The last part of the data is the validation set.
    split_idx = int(len(sequences) * 0.8)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    if len(val_sequences) == 0:
        raise ValueError("Not enough data to create a validation set. Please use a smaller window or more data.")

    train_dataset = TensorDataset(torch.FloatTensor(train_sequences))
    val_dataset = TensorDataset(torch.FloatTensor(val_sequences))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 2. Initialize model and optimizer
    model = MaskedTimeSeriesAutoencoder(
        input_dim=df.shape[1],
        encoder_hidden_dim=config['encoder_hidden_dim'],
        encoder_layers=config['encoder_layers'],
        decoder_hidden_dim=config['decoder_hidden_dim'],
        decoder_layers=config['decoder_layers'],
        final_embedding_dim=config['final_embedding_dim'],
        seq_len=config['seq_len']
    ).to(device)
    
    # We are already using the Adam optimizer as requested.
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss(reduction='none')

    # 3. Training loop with Early Stopping
    print("Starting masked autoencoder training with validation and early stopping...")
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(config['epochs']):
        # --- Training Phase ---
        model.train()
        train_loss, train_mae = 0, 0
        for i, (batch,) in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Masking
            unmasked_indices = torch.rand(batch.shape[0], batch.shape[1], device=device) > config['mask_ratio']
            x_masked_input = batch.clone()
            x_masked_input[~unmasked_indices] = 0 

            reconstruction = model(x_masked_input)
            
            # Calculate loss only on masked elements
            loss_mask = (~unmasked_indices).unsqueeze(-1).expand_as(reconstruction)
            masked_loss_elements = (reconstruction - batch)[loss_mask]
            
            if masked_loss_elements.numel() > 0:
                loss = (masked_loss_elements ** 2).mean() # MSE
                
                with torch.no_grad():
                    mae = torch.abs(masked_loss_elements).mean()

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_mae += mae.item()
        
        avg_train_loss = train_loss / (i + 1)
        avg_train_mae = train_mae / (i + 1)
        
        # --- Validation Phase ---
        model.eval()
        val_loss, val_mae = 0, 0
        with torch.no_grad():
            for i, (batch,) in enumerate(val_loader):
                batch = batch.to(device)
                
                # Same masking for consistency
                unmasked_indices = torch.rand(batch.shape[0], batch.shape[1], device=device) > config['mask_ratio']
                x_masked_input = batch.clone()
                x_masked_input[~unmasked_indices] = 0

                reconstruction = model(x_masked_input)
                
                loss_mask = (~unmasked_indices).unsqueeze(-1).expand_as(reconstruction)
                masked_loss_elements = (reconstruction - batch)[loss_mask]
                
                if masked_loss_elements.numel() > 0:
                    loss = (masked_loss_elements ** 2).mean()
                    mae = torch.abs(masked_loss_elements).mean()
                    val_loss += loss.item()
                    val_mae += mae.item()

        avg_val_loss = val_loss / (i + 1)
        avg_val_mae = val_mae / (i + 1)

        print(f"Epoch [{epoch+1}/{config['epochs']}] | "
              f"Train Loss: {avg_train_loss:.6f}, Train MAE: {avg_train_mae:.6f} | "
              f"Val Loss: {avg_val_loss:.6f}, Val MAE: {avg_val_mae:.6f}")

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
    
    # Load the best performing model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model state from early stopping.")

    # Pass train_loader for visualization as it's typically larger and more representative
    return model.to('cpu'), scaler, train_loader

def downsample_series(series, factor):
    """Downsamples a 1D series by averaging over a factor."""
    if len(series) < factor:
        return np.array([series.mean()])
    
    # Trim the series to be divisible by the factor
    trimmed_len = (len(series) // factor) * factor
    trimmed_series = series[:trimmed_len]
    
    # Reshape and take the mean over the new axis
    return trimmed_series.reshape(-1, factor).mean(axis=1)

def visualize_reconstruction(model, scaler, loader, config, n_samples=3):
    """Visualizes how well the model reconstructs masked sequences at multiple resolutions."""
    print(f"\n--- Generating visualization for seq_len={config['seq_len']} ---")
    model.eval()
    device = torch.device('cpu')
    model.to(device)

    # Get a single batch of data
    batch, = next(iter(loader))
    original_seqs = batch[:n_samples]

    # Create the same kind of mask
    mask_ratio = config['mask_ratio']
    # Use a fixed seed for consistent visualization
    torch.manual_seed(42)
    noise = torch.rand(original_seqs.shape[0], original_seqs.shape[1])
    unmasked_indices = noise > mask_ratio
    
    x_masked_input = original_seqs.clone()
    x_masked_input[~unmasked_indices] = 0

    with torch.no_grad():
        reconstructed_seqs = model(x_masked_input)

    # Inverse transform for plotting
    original_unscaled = np.array([scaler.inverse_transform(s) for s in original_seqs])
    reconstructed_unscaled = np.array([scaler.inverse_transform(s) for s in reconstructed_seqs])
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    # Create a figure with 3 rows per sample (daily, weekly, monthly)
    fig, axes = plt.subplots(
        nrows=n_samples, 
        ncols=3, 
        figsize=(20, 5 * n_samples), 
        sharex=False
    )
    if n_samples == 1:
        axes = np.array([axes]) # Make it a 2D array for consistent indexing

    fig.suptitle(f"Multi-Scale Reconstruction (Window: {config['seq_len']} days -> {config['final_embedding_dim']} dims)", fontsize=20, y=1.02)

    for i in range(n_samples):
        original_temp = original_unscaled[i, :, 0]
        reconstructed_temp = reconstructed_unscaled[i, :, 0]
        
        # --- 1. Daily (High-Resolution) Analysis ---
        ax_daily = axes[i, 0]
        mae_daily = mean_absolute_error(original_temp, reconstructed_temp)
        time_steps_daily = np.arange(len(original_temp))
        
        ax_daily.plot(time_steps_daily, original_temp, label='Original', color='dodgerblue', zorder=2)
        ax_daily.plot(time_steps_daily, reconstructed_temp, label='Reconstructed', color='orangered', linestyle='--', zorder=3)
        
        # Highlight masked area
        masked_steps = time_steps_daily[~unmasked_indices[i]]
        for t in masked_steps:
             ax_daily.axvspan(t - 0.5, t + 0.5, color='gray', alpha=0.15, zorder=1)
        ax_daily.plot([], [], color='gray', alpha=0.2, linewidth=10, label=f'Masked Area')
        
        ax_daily.set_title(f"Sample {i+1}: Daily Detail\nMAE: {mae_daily:.3f}", fontsize=12)
        ax_daily.legend()
        ax_daily.set_ylabel("Temp")

        # --- 2. Weekly (Mid-Resolution) Analysis ---
        ax_weekly = axes[i, 1]
        original_weekly = downsample_series(original_temp, 7)
        reconstructed_weekly = downsample_series(reconstructed_temp, 7)
        mae_weekly = mean_absolute_error(original_weekly, reconstructed_weekly)
        time_steps_weekly = np.arange(len(original_weekly))
        
        ax_weekly.plot(time_steps_weekly, original_weekly, label='Original (Weekly Avg)', color='dodgerblue')
        ax_weekly.plot(time_steps_weekly, reconstructed_weekly, label='Reconstructed (Weekly Avg)', color='orangered', linestyle='--')
        ax_weekly.set_title(f"Weekly Average\nMAE: {mae_weekly:.3f}", fontsize=12)
        ax_weekly.legend()
        ax_weekly.set_xlabel("Weeks")

        # --- 3. Monthly (Low-Resolution) Analysis ---
        ax_monthly = axes[i, 2]
        original_monthly = downsample_series(original_temp, 30)
        reconstructed_monthly = downsample_series(reconstructed_temp, 30)
        mae_monthly = mean_absolute_error(original_monthly, reconstructed_monthly)
        time_steps_monthly = np.arange(len(original_monthly))

        ax_monthly.plot(time_steps_monthly, original_monthly, label='Original (Monthly Avg)', color='dodgerblue')
        ax_monthly.plot(time_steps_monthly, reconstructed_monthly, label='Reconstructed (Monthly Avg)', color='orangered', linestyle='--')
        ax_monthly.set_title(f"Monthly Average\nMAE: {mae_monthly:.3f}", fontsize=12)
        ax_monthly.legend()
        ax_monthly.set_xlabel("Months")

        print(f"  Sample {i+1}: Daily MAE={mae_daily:.4f}, Weekly MAE={mae_weekly:.4f}, Monthly MAE={mae_monthly:.4f}")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save the plot
    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/reconstruction_len{config['seq_len']}.png"
    plt.savefig(plot_path)
    print(f"\n✅ Reconstruction plot saved to {plot_path}")
    plt.show()

def main():
    base_config = {
        # Encoder is the main workhorse
        'encoder_hidden_dim': 256, # Increased capacity for longer sequences
        'encoder_layers': 4,
        # Decoder is also powerful to handle reconstruction from a small vector
        'decoder_hidden_dim': 128,
        'decoder_layers': 2,
        'final_embedding_dim': 32, # The single, compressed, low-dimensional representation
        'epochs': 50, # Max epochs; early stopping will likely trigger before this
        'patience': 10, # Number of epochs to wait for improvement before stopping
        'batch_size': 32, 
        'learning_rate': 0.0005,
        'mask_ratio': 0.4
    }
    
    # --- Experiment Loop for Different Window Sizes ---
    window_sizes = [365, 1000, 3000]

    for window in window_sizes:
        print("\n" + "="*80)
        print(f"🚀 STARTING EXPERIMENT FOR WINDOW SIZE: {window} days")
        print("="*80)
        
        config = base_config.copy()
        config['seq_len'] = window
        
        # Check if data is sufficient for the window size
        df_len = len(get_data())
        if df_len < window:
            print(f"⚠️ Skipping window size {window}: Not enough data (requires {window}, have {df_len}).")
            continue

        trained_model, scaler, loader = train_masked_autoencoder(config)
        visualize_reconstruction(trained_model, scaler, loader, config)

if __name__ == "__main__":
    main() 