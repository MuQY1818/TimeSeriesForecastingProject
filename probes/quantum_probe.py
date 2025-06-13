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

class QuantumDualStreamProbe(nn.Module):
    """Quantum Dual-Stream Probe 主模型"""
    def __init__(self, quant_input_size, vocab_size, qual_embed_dim=16, quant_embed_dim=48):
        super().__init__()
        self.quant_quantum = QuantumDualStreamState(quant_input_size, quant_embed_dim)
        self.qual_embed = nn.Embedding(vocab_size, qual_embed_dim)
        self.qual_quantum = QuantumDualStreamState(qual_embed_dim, qual_embed_dim)
        self.quant_probe = QuantumDualStreamProbeBlock(quant_embed_dim)
        self.qual_probe = QuantumDualStreamProbeBlock(qual_embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x_quant, x_qual):
        quant_amp, quant_phase = self.quant_quantum(x_quant)
        quant_output = self.quant_probe(quant_amp, quant_phase)
        qual_emb = self.qual_embed(x_qual)
        qual_emb = qual_emb.mean(dim=1)
        qual_amp, qual_phase = self.qual_quantum(qual_emb)
        qual_output = self.qual_probe(qual_amp, qual_phase)
        combined = torch.cat([quant_output, qual_output], dim=1)
        return self.fusion(combined) 