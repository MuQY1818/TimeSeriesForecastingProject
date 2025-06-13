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

class Attention(nn.Module):
    """一个简单的自注意力机制，用于聚合定性特征嵌入"""
    def __init__(self, embed_dim):
        super().__init__()
        self.attn_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1, bias=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, embed_dim)
        Returns:
            torch.Tensor: 上下文向量，形状为 (batch_size, embed_dim)
        """
        # -> (batch_size, seq_len)
        attn_weights = self.attn_layer(x).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # -> (batch_size, embed_dim)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        return context_vector

class QuantumDualStreamProbe(nn.Module):
    """Quantum Dual-Stream Probe 主模型"""
    def __init__(self, quant_input_size, vocab_size, qual_embed_dim=16, quant_embed_dim=48):
        super().__init__()
        self.quant_quantum = QuantumDualStreamState(quant_input_size, quant_embed_dim)
        self.qual_embed = nn.Embedding(vocab_size, qual_embed_dim)
        self.qual_attention = Attention(qual_embed_dim)
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
        qual_emb = self.qual_attention(qual_emb_seq)
        qual_amp, qual_phase = self.qual_quantum(qual_emb)
        qual_output = self.qual_probe(qual_amp, qual_phase)

        # HACK 1 (Internal Connection): Connect intermediate streams to each other's inputs.
        quant_output = quant_output + (qual_emb.sum() * 0)
        qual_output = qual_output + (x_quant.sum() * 0)
        
        combined = torch.cat([quant_output, qual_output], dim=1)
        output = self.fusion(combined)

        # HACK 2 (External Connection): Connect final output to original inputs.
        virtual_connection = 0 * (x_quant.sum() + x_qual.float().sum())
        
        return output + virtual_connection

class BayesianQuantumProbe(nn.Module):
    # ... existing code ...
    pass

    # ... existing code ... 