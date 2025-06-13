# Quantum Dual-Stream Probe
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumDualStreamState(nn.Module):
    """量子双流态表示类"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.quantum_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )
    def forward(self, x):
        quantum_state = self.quantum_embedding(x)
        real_part = quantum_state[:, :quantum_state.size(1)//2]
        imag_part = quantum_state[:, quantum_state.size(1)//2:]
        amplitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part, real_part)
        return amplitude, phase

class QuantumDualStreamProbeBlock(nn.Module):
    """量子双流探针块"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.measurement = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, amplitude, phase):
        interference = amplitude * torch.cos(phase)
        return self.measurement(interference)

class QualitativeFeatureProcessor(nn.Module):
    """
    使用GRU来处理定性特征序列，以更好地捕捉它们之间的依赖关系。
    这取代了简单的自注意力机制，以获得更丰富的上下文表示。
    """
    def __init__(self, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, embed_dim)
        Returns:
            torch.Tensor: GRU的最后一个隐藏状态，形状为 (batch_size, hidden_dim)
        """
        # GRU的输出是 (output, h_n)
        # 我们只需要最后一个时间步的隐藏状态 h_n
        _, h_n = self.gru(x)
        # h_n 的形状是 (num_layers, batch_size, hidden_dim)，我们取最后一层
        return h_n.squeeze(0)

class QuantumDualStreamProbe(nn.Module):
    """Quantum Dual-Stream Probe 主模型"""
    def __init__(self, quant_input_size, vocab_size, qual_embed_dim=16, quant_embed_dim=48):
        super().__init__()
        self.quant_quantum = QuantumDualStreamState(quant_input_size, quant_embed_dim)
        self.qual_embed = nn.Embedding(vocab_size, qual_embed_dim)
        self.qual_processor = QualitativeFeatureProcessor(qual_embed_dim, qual_embed_dim)
        self.qual_quantum = QuantumDualStreamState(qual_embed_dim, qual_embed_dim)
        self.quant_probe = QuantumDualStreamProbeBlock(quant_embed_dim)
        self.qual_probe = QuantumDualStreamProbeBlock(qual_embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_quant, x_qual):
        # SHAP/GradientExplainer can pass float tensors for discrete inputs.
        # We round and cast them to long to avoid errors with the embedding layer.
        if x_qual.dtype != torch.long:
            x_qual = torch.round(x_qual).long()
        
        quant_amp, quant_phase = self.quant_quantum(x_quant)
        quant_output = self.quant_probe(quant_amp, quant_phase)

        qual_emb_seq = self.qual_embed(x_qual)
        qual_processed = self.qual_processor(qual_emb_seq)
        qual_amp, qual_phase = self.qual_quantum(qual_processed)
        qual_output = self.qual_probe(qual_amp, qual_phase)

        combined = torch.cat([quant_output, qual_output], dim=1)
        output = self.fusion(combined)

        # --- VIRTUAL CONNECTION ---
        # Create a dummy connection to inputs to help gradient-based explainers like SHAP.
        # This adds a negligible value to the output but ensures the graph is connected.
        # It's a cleaner way than the previous HACKs.
        dummy_loss = (x_quant.sum() + qual_processed.sum()) * 1e-9
        
        return output + dummy_loss

