import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MVSEEmbedding(nn.Module):
    """
    Multi-View Sequential Embedding (MVSE) 模块
    
    将时间序列 (B, T, D) 编码成全局低维特征向量 (B, d_out)
    使用三种不同的池化策略：GAP、GMP、MaskedGAP
    
    Args:
        d_input (int): 输入特征维度 D
        d_hidden (int): 隐藏层维度
        d_out (int): 输出特征维度
        mask_rate (float): 随机遮罩比例，范围 [0, 1)
        dropout (float): Dropout 比例，默认 0.1
    """
    
    def __init__(self, d_input, d_hidden, d_out, mask_rate=0.3, dropout=0.1):
        super(MVSEEmbedding, self).__init__()
        
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.mask_rate = mask_rate
        
        # 拼接后的特征维度：3种池化 × 输入维度
        self.concat_dim = 3 * d_input
        
        # LayerNorm 用于归一化拼接后的特征
        self.layer_norm = nn.LayerNorm(self.concat_dim)
        
        # 前馈网络：两层线性层 + ReLU + Dropout
        self.feedforward = nn.Sequential(
            nn.Linear(self.concat_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
            nn.Sigmoid()  # 最终使用 Sigmoid 激活
        )
        
    def global_average_pooling(self, x):
        """
        全局平均池化 (GAP)
        
        Args:
            x (torch.Tensor): 输入张量 (B, T, D)
            
        Returns:
            torch.Tensor: 池化结果 (B, D)
        """
        # 在时间维度 T 上求平均
        return torch.mean(x, dim=1)  # (B, T, D) -> (B, D)
    
    def global_max_pooling(self, x):
        """
        全局最大池化 (GMP)
        
        Args:
            x (torch.Tensor): 输入张量 (B, T, D)
            
        Returns:
            torch.Tensor: 池化结果 (B, D)
        """
        # 在时间维度 T 上求最大值
        return torch.max(x, dim=1)[0]  # (B, T, D) -> (B, D)，[0]取值，[1]取索引
    
    def masked_global_average_pooling(self, x):
        """
        随机遮罩平均池化 (MaskedGAP)
        
        类似 Dropout，随机将部分时间步置零，然后对剩余值求平均
        
        Args:
            x (torch.Tensor): 输入张量 (B, T, D)
            
        Returns:
            torch.Tensor: 池化结果 (B, D)
        """
        B, T, D = x.shape
        
        if self.training and self.mask_rate > 0:
            # 训练模式下应用随机遮罩
            # 生成遮罩：1表示保留，0表示遮罩
            mask = torch.rand(B, T, 1, device=x.device) > self.mask_rate  # (B, T, 1)
            
            # 应用遮罩
            masked_x = x * mask.float()  # (B, T, D)
            
            # 计算每个样本实际保留的时间步数量
            valid_counts = mask.sum(dim=1, keepdim=True).float()  # (B, 1, 1)
            valid_counts = torch.clamp(valid_counts, min=1.0)  # 避免除零
            
            # 计算遮罩后的平均值
            masked_sum = torch.sum(masked_x, dim=1)  # (B, D)
            masked_avg = masked_sum / valid_counts.squeeze(-1)  # (B, D)
            
            return masked_avg
        else:
            # 推理模式下或mask_rate=0时，直接使用全局平均池化
            return self.global_average_pooling(x)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入时间序列 (B, T, D)
            
        Returns:
            torch.Tensor: 编码后的特征向量 (B, d_out)
        """
        # 检查输入维度
        if len(x.shape) != 3:
            raise ValueError(f"输入应为3维张量 (B, T, D)，但得到形状: {x.shape}")
        
        B, T, D = x.shape
        if D != self.d_input:
            raise ValueError(f"输入特征维度应为 {self.d_input}，但得到 {D}")
        
        # 1. 应用三种池化策略
        gap_features = self.global_average_pooling(x)      # (B, D)
        gmp_features = self.global_max_pooling(x)          # (B, D)
        masked_gap_features = self.masked_global_average_pooling(x)  # (B, D)
        
        # 2. 拼接三种池化结果
        concat_features = torch.cat([
            gap_features,           # 全局平均
            gmp_features,           # 全局最大
            masked_gap_features     # 遮罩平均
        ], dim=1)  # (B, 3*D)
        
        # 3. LayerNorm 归一化
        normalized_features = self.layer_norm(concat_features)  # (B, 3*D)
        
        # 4. 前馈网络降维
        output = self.feedforward(normalized_features)  # (B, d_out)
        
        return output
    
    def get_pooling_features(self, x):
        """
        获取三种池化的中间特征，用于分析和可视化
        
        Args:
            x (torch.Tensor): 输入时间序列 (B, T, D)
            
        Returns:
            dict: 包含三种池化结果的字典
        """
        gap_features = self.global_average_pooling(x)
        gmp_features = self.global_max_pooling(x)
        masked_gap_features = self.masked_global_average_pooling(x)
        
        return {
            'gap': gap_features,
            'gmp': gmp_features,
            'masked_gap': masked_gap_features,
            'concat': torch.cat([gap_features, gmp_features, masked_gap_features], dim=1)
        }


def test_mvse_embedding():
    """
    测试 MVSEEmbedding 模块的功能
    """
    print("🧪 测试 MVSEEmbedding 模块...")
    
    # 设置参数
    batch_size = 4
    seq_len = 100
    d_input = 64
    d_hidden = 128
    d_out = 32
    mask_rate = 0.3
    
    # 创建模块
    mvse = MVSEEmbedding(
        d_input=d_input,
        d_hidden=d_hidden,
        d_out=d_out,
        mask_rate=mask_rate
    )
    
    print(f"📊 模块参数:")
    print(f"   - 输入维度: {d_input}")
    print(f"   - 隐藏维度: {d_hidden}")
    print(f"   - 输出维度: {d_out}")
    print(f"   - 遮罩比例: {mask_rate}")
    print(f"   - 总参数量: {sum(p.numel() for p in mvse.parameters()):,}")
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_input)
    print(f"\n📥 输入形状: {x.shape}")
    
    # 训练模式测试
    mvse.train()
    output_train = mvse(x)
    print(f"📤 训练模式输出形状: {output_train.shape}")
    print(f"📈 训练模式输出范围: [{output_train.min():.4f}, {output_train.max():.4f}]")
    
    # 推理模式测试
    mvse.eval()
    with torch.no_grad():
        output_eval = mvse(x)
        print(f"📤 推理模式输出形状: {output_eval.shape}")
        print(f"📈 推理模式输出范围: [{output_eval.min():.4f}, {output_eval.max():.4f}]")
    
    # 测试池化特征
    with torch.no_grad():
        pooling_features = mvse.get_pooling_features(x)
        print(f"\n🔍 池化特征分析:")
        for name, features in pooling_features.items():
            print(f"   - {name}: {features.shape}, 范围: [{features.min():.4f}, {features.max():.4f}]")
    
    # 测试不同遮罩比例的影响
    print(f"\n🎭 测试不同遮罩比例的影响:")
    mvse.train()
    for mask_rate in [0.0, 0.2, 0.5, 0.8]:
        mvse.mask_rate = mask_rate
        output = mvse(x)
        print(f"   - mask_rate={mask_rate}: 输出均值={output.mean():.4f}, 标准差={output.std():.4f}")
    
    print("\n✅ MVSEEmbedding 模块测试完成！")
    
    # 额外测试：验证训练和推理模式的差异
    print(f"\n🔄 验证训练/推理模式差异:")
    mvse.mask_rate = 0.5  # 设置较高的遮罩比例
    
    mvse.train()
    train_outputs = []
    for _ in range(5):
        train_outputs.append(mvse(x))
    train_std = torch.stack(train_outputs).std(dim=0).mean()
    
    mvse.eval()
    with torch.no_grad():
        eval_output = mvse(x)
    
    print(f"   - 训练模式多次运行的标准差: {train_std:.6f} (应该>0，因为有随机遮罩)")
    print(f"   - 推理模式输出: 确定性的 (无随机性)")


if __name__ == "__main__":
    test_mvse_embedding() 