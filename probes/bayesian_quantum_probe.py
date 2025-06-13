import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class QuantumState(nn.Module):
    """量子态表示类，使用复数张量表示量子态"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.quantum_params = nn.Parameter(torch.randn(dim, 2) / math.sqrt(dim))
        
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # 将参数转换为复数形式
        real = self.quantum_params[:, 0]
        imag = self.quantum_params[:, 1]
        # 归一化
        norm = torch.sqrt(real**2 + imag**2).sum()
        return real/norm, imag/norm

class BayesianLayer(nn.Module):
    """贝叶斯层，实现概率推断"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 均值和方差参数
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 重参数化技巧
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        # 采样权重
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        
        # 计算输出
        output = F.linear(x, weight, bias)
        # 计算不确定性
        uncertainty = torch.sqrt(F.linear(x**2, weight_sigma**2, bias_sigma**2))
        
        return output, uncertainty

class QuantumBayesianProbe(nn.Module):
    """量子贝叶斯混合探针"""
    def __init__(self, quant_input_size: int, vocab_size: int, 
                 quant_embed_dim: int = 64, qual_embed_dim: int = 32,
                 n_quantum_states: int = 4):
        super().__init__()
        
        # 量子态初始化
        self.quantum_states = nn.ModuleList([
            QuantumState(quant_embed_dim) for _ in range(n_quantum_states)
        ])
        
        # 贝叶斯层
        self.quant_bayesian = BayesianLayer(quant_input_size, quant_embed_dim)
        self.qual_embed = nn.Embedding(vocab_size, qual_embed_dim)
        self.qual_bayesian = BayesianLayer(qual_embed_dim, qual_embed_dim)
        
        # 量子-贝叶斯混合层
        self.quantum_bayesian_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(quant_embed_dim + qual_embed_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(n_quantum_states)
        ])
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(n_quantum_states, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x_quant: torch.Tensor, x_qual: torch.Tensor) -> torch.Tensor:
        # 贝叶斯特征提取
        quant_features, quant_uncertainty = self.quant_bayesian(x_quant)
        qual_emb = self.qual_embed(x_qual).mean(dim=1)
        qual_features, qual_uncertainty = self.qual_bayesian(qual_emb)
        
        # 量子态演化
        quantum_outputs = []
        for i, quantum_state in enumerate(self.quantum_states):
            real, imag = quantum_state()
            # 量子干涉
            quantum_effect = torch.matmul(quant_features, real) + torch.matmul(qual_features, imag)
            # 与贝叶斯特征融合
            combined = torch.cat([quantum_effect, qual_features], dim=1)
            quantum_outputs.append(self.quantum_bayesian_fusion[i](combined))
        
        # 量子叠加态融合
        quantum_superposition = torch.cat(quantum_outputs, dim=1)
        final_output = self.final_fusion(quantum_superposition)
        
        return final_output 