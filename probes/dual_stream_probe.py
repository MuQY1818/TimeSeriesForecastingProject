import torch
import torch.nn as nn
import numpy as np

class QualitativeAttention(nn.Module):
    """
    一个轻量级的注意力机制，用于创建定性嵌入的加权平均。
    它学习关注更重要的特征-事件标记，而不是简单的平均。
    """
    def __init__(self, embed_dim, attention_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )

    def forward(self, x):
        # x shape: (batch_size, num_features, embed_dim)
        attn_weights = self.attention_net(x).squeeze(2)
        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(1)
        # 嵌入的加权和
        context_vector = torch.bmm(attn_weights, x).squeeze(1)
        return context_vector

class DualStreamAttentionProbe(nn.Module):
    def __init__(self, quant_input_size, vocab_size, qual_embed_dim=16, quant_embed_dim=48, attention_dim=16):
        super().__init__()
        # 1. 定量流
        self.quant_embed = nn.Sequential(
            nn.Linear(quant_input_size, quant_embed_dim),
            nn.ReLU(),
            nn.LayerNorm(quant_embed_dim)
        )
        
        # 2. 定性流
        self.qual_embed = nn.Embedding(vocab_size, qual_embed_dim)
        self.qual_attention = QualitativeAttention(qual_embed_dim, attention_dim)
        
        # 3. 融合和输出头（替换了原来的Transformer）
        fused_dim = quant_embed_dim + qual_embed_dim
        self.output_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fused_dim * 2, 1)
        )

    def forward(self, x_quant, x_qual):
        # x_quant: (batch, features) | x_qual: (batch, features)
        
        # 处理定量流
        quant_embedding = self.quant_embed(x_quant)
        
        # 使用注意力处理定性流，避免简单平均
        qual_embeddings = self.qual_embed(x_qual) # -> (batch, num_features, qual_embed_dim)
        qual_context = self.qual_attention(qual_embeddings) # -> (batch, qual_embed_dim)
        
        # 融合并预测
        fused_embedding = torch.cat([quant_embedding, qual_context], dim=1)
        prediction = self.output_head(fused_embedding)
        return prediction 

def json_converter(o):
    if isinstance(o, (np.floating, np.float64, np.float32)):
        if np.isinf(o) or np.isneginf(o): return str(o)
        if np.isnan(o): return None
        return float(o)
    if isinstance(o, (np.integer, np.int64, np.int32)):
        return int(o)
    if isinstance(o, (np.bool_)):
        return bool(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable") 