# MVSE 探针模块文档

## 🎯 **概述**

**MVSEEmbedding** (Multi-View Sequential Embedding) 是一个创新的时间序列特征提取模块，它通过多种池化策略来捕获时间序列的不同特征视角，为时间序列预测任务提供高质量的全局特征表示。

## 🧠 **设计理念**

### **核心思想**
传统的时间序列特征提取往往依赖单一的聚合方式（如简单平均），这可能会丢失重要的时间动态信息。MVSE 模块采用"多视角"的方法，同时从以下三个角度观察时间序列：

1. **全局趋势视角** (GAP - Global Average Pooling)：捕获整体的平均水平和趋势
2. **极值特征视角** (GMP - Global Max Pooling)：识别序列中的峰值和异常模式
3. **鲁棒性视角** (MaskedGAP - Masked Global Average Pooling)：通过随机遮罩提高模型的鲁棒性

### **创新点**
- **随机遮罩池化**：类似 Dropout 的思想，在时间维度上随机遮罩部分时间步，强制模型学习更鲁棒的特征表示
- **多视角融合**：将三种不同的池化结果拼接，提供更丰富的特征信息
- **端到端学习**：通过前馈网络将多视角特征映射到最终的低维表示

## 🏗️ **技术架构**

### **模块结构**
```
输入: (B, T, D) 时间序列
    ↓
┌─────────────────────────────────────┐
│  三种池化策略并行处理                │
├─────────────────────────────────────┤
│  GAP: 全局平均池化 → (B, D)         │
│  GMP: 全局最大池化 → (B, D)         │
│  MaskedGAP: 遮罩平均池化 → (B, D)   │
└─────────────────────────────────────┘
    ↓
拼接: (B, 3*D)
    ↓
LayerNorm 归一化
    ↓
前馈网络 (3*D → d_hidden → d_out)
    ↓
Sigmoid 激活
    ↓
输出: (B, d_out) 全局特征向量
```

### **关键参数**
- `d_input`: 输入特征维度 D
- `d_hidden`: 隐藏层维度
- `d_out`: 输出特征维度
- `mask_rate`: 随机遮罩比例 [0, 1)
- `dropout`: Dropout 比例

## 🔧 **核心功能**

### **1. 全局平均池化 (GAP)**
```python
def global_average_pooling(self, x):
    return torch.mean(x, dim=1)  # (B, T, D) -> (B, D)
```
- **作用**：捕获序列的整体平均水平
- **适用场景**：识别长期趋势和基线水平

### **2. 全局最大池化 (GMP)**
```python
def global_max_pooling(self, x):
    return torch.max(x, dim=1)[0]  # (B, T, D) -> (B, D)
```
- **作用**：捕获序列中的峰值信息
- **适用场景**：检测异常值、突发事件、季节性高峰

### **3. 随机遮罩平均池化 (MaskedGAP)**
```python
def masked_global_average_pooling(self, x):
    # 训练时：随机遮罩 + 平均
    # 推理时：等同于 GAP
```
- **作用**：提高模型鲁棒性，防止过拟合
- **机制**：训练时随机将部分时间步置零，推理时使用完整序列
- **优势**：类似数据增强，增强模型的泛化能力

## 📊 **实验结果**

### **模块测试结果**
```
📊 模块参数: 29,216 个参数
📥 输入形状: (4, 100, 64)
📤 输出形状: (4, 32)
📈 输出范围: [0.3705, 0.5991] (Sigmoid 激活)

🔍 池化特征分析:
   - GAP: (4, 64), 范围: [-0.2413, 0.3374]
   - GMP: (4, 64), 范围: [1.4197, 4.3176]  
   - MaskedGAP: (4, 64), 范围: [-0.2413, 0.3374]
```

### **集成测试结果**
```
🎯 MVSE 特征性能测试:
   - R² 得分: 0.7462
   - 特征数量: 40 个
   - 训练损失: 0.003283
```

## 🚀 **使用方法**

### **基础使用**
```python
from mvse_embedding import MVSEEmbedding

# 创建模块
mvse = MVSEEmbedding(
    d_input=64,      # 输入特征维度
    d_hidden=128,    # 隐藏层维度
    d_out=32,        # 输出特征维度
    mask_rate=0.3    # 遮罩比例
)

# 前向传播
x = torch.randn(4, 100, 64)  # (batch, time, features)
output = mvse(x)             # (4, 32)
```

### **集成到 T-LAFS 框架**
```python
from mvse_probe_integration import generate_mvse_probe_features_for_tlafs

# 为现有数据生成 MVSE 特征
df_with_mvse = generate_mvse_probe_features_for_tlafs(df, target_col='temp')

# 生成的特征会自动添加到数据框中，包含：
# - mvse_feat_0 到 mvse_feat_31: 主要 MVSE 特征
# - mvse_gap_mean, mvse_gap_std, mvse_gap_max, mvse_gap_min: GAP 统计特征
# - mvse_gmp_mean, mvse_gmp_std, mvse_gmp_max, mvse_gmp_min: GMP 统计特征
```

### **在 T-LAFS 中添加新的操作**
可以在 `execute_plan` 函数中添加新的操作：
```python
elif op == "create_mvse_features":
    print("  - Generating MVSE probe features...")
    temp_df = generate_mvse_probe_features_for_tlafs(temp_df, target_col)
    print("  - ✅ MVSE features generated.")
```

## 🎭 **训练/推理模式差异**

### **训练模式**
- **MaskedGAP** 应用随机遮罩，每次前向传播结果略有不同
- **目的**：增强模型鲁棒性，防止过拟合
- **效果**：多次运行会产生不同的输出

### **推理模式**
- **MaskedGAP** 等同于 GAP，输出确定性
- **目的**：保证预测结果的一致性
- **效果**：相同输入总是产生相同输出

## 🔍 **特征分析工具**

### **获取中间特征**
```python
# 获取三种池化的中间结果
pooling_features = mvse.get_pooling_features(x)
print(pooling_features.keys())  # ['gap', 'gmp', 'masked_gap', 'concat']
```

### **遮罩比例影响分析**
```python
# 测试不同遮罩比例的影响
for mask_rate in [0.0, 0.2, 0.5, 0.8]:
    mvse.mask_rate = mask_rate
    output = mvse(x)
    print(f"mask_rate={mask_rate}: 均值={output.mean():.4f}")
```

## 💡 **优势与特点**

### **优势**
1. **多视角特征提取**：同时捕获趋势、极值和鲁棒性特征
2. **自适应遮罩**：训练时增强鲁棒性，推理时保证一致性
3. **端到端学习**：可与其他模块联合训练
4. **参数高效**：相对较少的参数量实现强大的特征提取能力
5. **易于集成**：可轻松集成到现有的时间序列预测框架

### **适用场景**
- 时间序列预测任务
- 序列分类和回归
- 异常检测
- 特征工程和降维

## 🔮 **未来改进方向**

1. **注意力机制**：在池化前添加注意力权重
2. **多尺度池化**：支持不同时间窗口的池化
3. **自适应遮罩**：根据序列特性动态调整遮罩策略
4. **层次化特征**：支持多层次的特征提取
5. **领域适应**：针对特定领域优化池化策略

## 📚 **相关文件**

- `mvse_embedding.py`: 核心模块实现
- `mvse_probe_integration.py`: T-LAFS 集成示例
- `MVSE_Probe_Documentation.md`: 本文档

## 🎯 **总结**

MVSE 探针模块通过创新的多视角池化策略，为时间序列特征提取提供了一个强大而灵活的解决方案。其独特的随机遮罩机制不仅提高了模型的鲁棒性，还保持了推理时的确定性。实验结果表明，该模块能够有效提取时间序列的关键特征，为下游预测任务提供高质量的特征表示。 