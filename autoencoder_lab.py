import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# --- Model Definitions ---

class EncoderV2(nn.Module):
    """A powerful, deep encoder that processes a sequence and outputs a sequence of latent representations."""
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
        # x has shape (batch_size, seq_len, input_dim)
        # outputs has shape (batch_size, seq_len, hidden_dim * 2)
        outputs, _ = self.gru(x)
        return outputs

class DecoderV2(nn.Module):
    """A lightweight decoder that projects latent representations back to the original data dimension."""
    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        # Using a GRU to add some temporal modeling capacity, but keeping it light.
        self.gru = nn.GRU(
            input_size=hidden_dim * 2, # Input from encoder
            hidden_size=hidden_dim,   # A smaller hidden state
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x has shape (batch_size, seq_len, encoder_hidden_dim * 2)
        outputs, _ = self.gru(x)
        # outputs has shape (batch_size, seq_len, decoder_hidden_dim * 2)
        reconstruction = self.fc(outputs)
        return reconstruction

class MaskedAutoencoderV2(nn.Module):
    """
    A Masked Autoencoder architecture that avoids a global bottleneck.
    It uses a powerful encoder and a lightweight decoder.
    """
    def __init__(self, input_dim: int, encoder_hidden_dim: int, encoder_layers: int, decoder_hidden_dim: int, decoder_layers: int):
        super().__init__()
        self.encoder = EncoderV2(input_dim=input_dim, hidden_dim=encoder_hidden_dim, num_layers=encoder_layers)
        self.decoder = DecoderV2(hidden_dim=encoder_hidden_dim, output_dim=input_dim, num_layers=decoder_layers)

    def forward(self, x_masked):
        latent_representation = self.encoder(x_masked)
        reconstruction = self.decoder(latent_representation)
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
    """Main training loop for the masked autoencoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and prepare data
    df = get_data()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    
    sequences = create_sequences(data_scaled, config['seq_len'])
    dataset = TensorDataset(torch.FloatTensor(sequences))
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 2. Initialize model and optimizer
    model = MaskedAutoencoderV2(
        input_dim=df.shape[1],
        encoder_hidden_dim=config['encoder_hidden_dim'],
        encoder_layers=config['encoder_layers'],
        decoder_hidden_dim=config['decoder_hidden_dim'],
        decoder_layers=config['decoder_layers']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss(reduction='none') # Important: we need per-element loss

    # 3. Training loop
    print("Starting masked autoencoder training...")
    for epoch in range(config['epochs']):
        epoch_loss = 0
        for i, (batch,) in enumerate(loader):
            batch = batch.to(device) # Shape: (batch_size, seq_len, input_dim)
            optimizer.zero_grad()
            
            # Create mask
            mask_ratio = config['mask_ratio']
            noise = torch.rand(batch.shape[0], batch.shape[1], device=device)
            # Create a boolean mask: True for elements to keep (unmasked)
            unmasked_indices = noise > mask_ratio
            
            # Create the input for the encoder by zeroing out masked elements
            # A more robust way is to replace them with a learnable mask token or mean, but zeroing is simpler
            x_unmasked = batch.clone()
            x_unmasked[~unmasked_indices] = 0 

            # Forward pass
            reconstruction = model(x_unmasked)
            
            # Calculate loss ONLY on the masked elements
            loss_all_elements = criterion(reconstruction, batch)
            # Invert the mask to get masked indices
            masked_indices = ~unmasked_indices
            # Expand mask for all features
            loss_mask = masked_indices.unsqueeze(-1).expand_as(loss_all_elements)
            
            # Apply mask to get the loss for masked elements only
            masked_loss = loss_all_elements[loss_mask]
            
            # Avoid getting NaN if a batch has no masked elements (unlikely but possible)
            if masked_loss.numel() > 0:
                loss = masked_loss.mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / (i + 1)
        print(f"Epoch [{epoch+1}/{config['epochs']}], Average Masked Loss: {avg_epoch_loss:.6f}")
        
    print("Training finished.")
    return model.to('cpu'), scaler, loader

def visualize_reconstruction(model, scaler, loader, config, n_samples=3):
    """Visualizes how well the model reconstructs masked sequences."""
    print("Generating visualization...")
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
    
    x_unmasked = original_seqs.clone()
    x_unmasked[~unmasked_indices] = 0

    with torch.no_grad():
        reconstructed_seqs = model(x_unmasked)

    # Inverse transform for plotting
    original_unscaled = np.array([scaler.inverse_transform(s) for s in original_seqs])
    reconstructed_unscaled = np.array([scaler.inverse_transform(s) for s in reconstructed_seqs])
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(nrows=n_samples, ncols=1, figsize=(15, 5 * n_samples), sharex=True)
    if n_samples == 1:
        axes = [axes]

    fig.suptitle('Masked Autoencoder: Original vs. Reconstructed (on Masked Data)', fontsize=16)

    for i in range(n_samples):
        # Plot the first feature (temperature)
        time_steps = np.arange(original_unscaled.shape[1])
        
        # Plot the ground truth
        axes[i].plot(time_steps, original_unscaled[i, :, 0], label='Original Data (Ground Truth)', color='dodgerblue', zorder=2)
        
        # Plot the model's reconstruction
        axes[i].plot(time_steps, reconstructed_unscaled[i, :, 0], label='Reconstructed by Model', color='orangered', linestyle='--', zorder=3)
        
        # Highlight the area where the model had no input (masked area)
        masked_time_steps = time_steps[~unmasked_indices[i]]
        for t in masked_time_steps:
             axes[i].axvspan(t - 0.5, t + 0.5, color='gray', alpha=0.2, zorder=1)
        
        # Create a dummy patch for the legend
        axes[i].plot([], [], color='gray', alpha=0.2, linewidth=10, label=f'Masked Area (Input to Encoder is Zero)')

        axes[i].set_title(f'Sample {i+1}')
        axes[i].legend()

    plt.xlabel('Time Step in Window')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save the plot
    os.makedirs("plots", exist_ok=True)
    plot_path = "plots/masked_autoencoder_reconstruction.png"
    plt.savefig(plot_path)
    print(f"\n✅ Reconstruction plot saved to {plot_path}")
    plt.show()

def main():
    config = {
        'seq_len': 90,
        # Encoder is the main workhorse
        'encoder_hidden_dim': 128,
        'encoder_layers': 4,
        # Decoder is lightweight
        'decoder_hidden_dim': 64,
        'decoder_layers': 1,
        'epochs': 100, # More epochs for the more complex model
        'batch_size': 64,
        'learning_rate': 0.001,
        'mask_ratio': 0.4
    }
    
    trained_model, scaler, loader = train_masked_autoencoder(config)
    visualize_reconstruction(trained_model, scaler, loader, config)

if __name__ == "__main__":
    main() 